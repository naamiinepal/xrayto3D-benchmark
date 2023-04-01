"""custom pytorch-lightning callbacks for saving model prediction
and evaluating metrics."""
import csv
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pytorch_lightning as pl
from monai.data.nifti_saver import NiftiSaver
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.metrics.surface_distance import SurfaceDistanceMetric
from pytorch_lightning.callbacks import BasePredictionWriter
from surface_distance.metrics import (
    compute_surface_distances,
    compute_surface_overlap_at_tolerance,
)
from typing_extensions import Literal

from .io_utils import get_nifti_stem, to_numpy


class NiftiPredictionWriter(BasePredictionWriter):
    """Save model prediction as nifti.
    Inherits from pytorch-lightning callbacks for Writing model prediction"""

    def __init__(
        self,
        output_dir,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
        save_pred=True,
        save_gt=True,
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_pred = save_pred
        self.save_gt = save_gt

        self.pred_nifti_saver = NiftiSaver(
            output_dir=self.output_dir,
            output_postfix="pred",
            resample=True,
            dtype=np.int16,
            separate_folder=False,
        )

        self.gt_nifti_saver = NiftiSaver(
            output_dir=self.output_dir,
            output_postfix="gt",
            resample=True,
            dtype=np.int16,
            separate_folder=False,
        )

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if self.save_pred:
            self.pred_nifti_saver.save_batch(
                prediction["pred"], prediction["seg_meta_dict"]
            )
        if self.save_gt:
            self.gt_nifti_saver.save_batch(
                prediction["gt"], prediction["seg_meta_dict"]
            )


class MetricsLogger(BasePredictionWriter):
    """evaluate various metrics from model prediction
    and save to log file"""

    def __init__(
        self,
        output_dir,
        voxel_spacing,
        nsd_tolerance=1.5,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(write_interval)
        Path(output_dir).mkdir(exist_ok=True, parents=False)
        self.filestream_writer = csv.writer(
            open(Path(output_dir) / "metric-log.csv", "w")
        )
        header = ["subject-id", "DSC", "ASD", "HD95", "NSD"]
        self.filestream_writer.writerow(header)
        # metric
        self.DSC = DiceMetric()
        self.ASD = SurfaceDistanceMetric()
        self.HD95 = HausdorffDistanceMetric(percentile=95)
        voxel_spacing = (voxel_spacing,) * 3  # resolution of the voxel grid
        self.NSD = lambda pred, gt: compute_surface_overlap_at_tolerance(
            compute_surface_distances(
                to_numpy(gt).astype(bool).squeeze(),
                to_numpy(pred).astype(bool).squeeze(),
                voxel_spacing,
            ),
            nsd_tolerance,
        )[0]

    def get_filename(self, prediction: Any):
        """this will be column name identifier for each row"""
        return [
            get_nifti_stem(p) for p in prediction["seg_meta_dict"]["filename_or_obj"]
        ]

    def write_on_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        prediction: Any,
        batch_indices: Optional[Sequence[int]],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pred = prediction["pred"]
        gt = prediction["gt"]
        subjects = self.get_filename(prediction)
        dsc = to_numpy(self.DSC(pred, gt)).flatten().tolist()
        asd = to_numpy(self.ASD(pred, gt)).flatten().tolist()
        hd95 = to_numpy(self.HD95(pred, gt)).flatten().tolist()
        nsd = [self.NSD(p, g) for p, g in zip(pred, gt)]
        for row in zip(subjects, dsc, asd, hd95, nsd):
            self.filestream_writer.writerow(
                [
                    f"{item:.2f}"
                    if isinstance(item, (np.ndarray, np.float64, float))
                    else item
                    for item in row
                ]
            )
