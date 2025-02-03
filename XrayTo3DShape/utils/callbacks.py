# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""custom pytorch-lightning callbacks for saving model prediction
and evaluating metrics."""
import copy
import csv
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
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
        image_size=64,
        resolution=1.5,
        save_input=True
    ) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_pred = save_pred
        self.save_gt = save_gt
        self.save_input = save_input
        self.image_size = image_size
        self.resolution = resolution

        self.pred_nifti_saver = NiftiSaver(
            output_dir=self.output_dir,
            output_postfix="pred",
            resample=False,
            dtype=np.int16,
            separate_folder=False,
        )

        self.gt_nifti_saver = NiftiSaver(
            output_dir=self.output_dir,
            output_postfix="gt",
            resample=False,
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
        # rewrite metadict to reflect the actual shape and orientation of the volume
        batch_size = len(prediction["seg_meta_dict"]["filename_or_obj"])
        original_meta_dict = prediction["seg_meta_dict"]
        metadict = copy.deepcopy(original_meta_dict)
        # metadict["filename_or_obj"] = prediction["seg_meta_dict"]["filename_or_obj"]
        metadict["spatial_shape"] = torch.from_numpy(
            np.asarray(
                [
                    [self.image_size, self.image_size, self.image_size],
                ]
                * batch_size
            )
        )
        metadict["space"] = [
            "PIL",
        ] * batch_size
        metadict["affine"] = torch.from_numpy(
            np.asarray(
                [
                    [
                        [0, 0, self.resolution, 0],
                        [-self.resolution, 0, 0, 0],
                        [0, -self.resolution, 0, 0],
                        [0, 0, 0, 1],
                    ],
                ]
                * batch_size
            )
        )
        # save the origin information too, which resides in the last column of the `affine matrix`

        metadict["affine"][:, :, -1] = torch.tensor(
            original_meta_dict["affine"][:, :, -1]
        )
        if self.save_pred:
            self.pred_nifti_saver.save_batch(prediction["pred"], metadict)
        if self.save_gt:
            self.gt_nifti_saver.save_batch(prediction["gt"], metadict)


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
        Path(output_dir).mkdir(exist_ok=True, parents=True)
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


class AnglePerturbationMetricsLogger(MetricsLogger):
    """additionally record the angle perturbation in addition to metrics"""

    def __init__(
        self,
        output_dir,
        voxel_spacing,
        nsd_tolerance=1.5,
        write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",
    ) -> None:
        super().__init__(output_dir, voxel_spacing, nsd_tolerance, write_interval)

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
        print(prediction["seg_meta_dict"]["original_affine"])
        print(prediction["seg_meta_dict"]["filename_or_obj"])
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
