from monai.data.nifti_saver import NiftiSaver
from pytorch_lightning.callbacks import BasePredictionWriter
import pytorch_lightning as pl
from typing import Any,Sequence,Optional
import numpy as np
import csv
from typing_extensions import Literal
from monai.metrics.meandice import DiceMetric
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.surface_distance import SurfaceDistanceMetric
from surface_distance.metrics import compute_surface_distances, compute_surface_overlap_at_tolerance
from .io_utils import to_numpy,get_nifti_stem
from pathlib import Path
import torch

class NiftiPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch",save_pred=True,save_gt=True) -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.save_pred = save_pred
        self.save_gt = save_gt
        
        self.pred_nifti_saver = NiftiSaver(output_dir=self.output_dir,output_postfix='pred',resample=True,dtype=np.int16,separate_folder=False)
        
        self.gt_nifti_saver = NiftiSaver(output_dir=self.output_dir,output_postfix='gt',resample=True,dtype=np.int16,separate_folder=False)

    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.save_pred:
            self.pred_nifti_saver.save_batch(prediction['pred'],prediction['seg_meta_dict'])
        if self.save_gt:
            self.gt_nifti_saver.save_batch(prediction['gt'],prediction['seg_meta_dict'])

class MetricsLogger(BasePredictionWriter):
    def __init__(self, output_dir,voxel_spacing,nsd_tolerance=1,write_interval: Literal["batch", "epoch", "batch_and_epoch"] = "batch") -> None:
        super().__init__(write_interval)
        self.filestream_writer = csv.writer(open(Path(output_dir)/'metric-log.csv','w'))
        header = ['subject-id', 'DSC', 'ASD','HD95',  'NSD']
        self.filestream_writer.writerow(header)
        # metric
        self.DSC = DiceMetric()
        self.ASD = SurfaceDistanceMetric()
        self.HD95 = HausdorffDistanceMetric(percentile=95)
        voxel_spacing = (voxel_spacing,)*3 # resolution of the voxel grid
        self.NSD = lambda pred,gt : compute_surface_overlap_at_tolerance(compute_surface_distances(to_numpy(gt).astype(bool).squeeze(),
                                                                                                       to_numpy(pred).astype(bool).squeeze(), voxel_spacing),
                                                                                                       nsd_tolerance)[0]
    def get_filename(self,prediction:Any):
        return [get_nifti_stem(p).split('_')[0] for p in prediction['seg_meta_dict']['filename_or_obj']]
    
    def post_eval_transform(self,x:torch.Tensor): return x.cpu().numpy().squeeze()

    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pred = prediction['pred']
        gt = prediction['gt']
        subjects = self.get_filename(prediction)
        dsc = to_numpy(self.DSC(pred,gt)).flatten().tolist()
        asd = to_numpy(self.ASD(pred,gt)).flatten().tolist()
        hd95 = to_numpy(self.HD95(pred,gt)).flatten().tolist()
        nsd = [self.NSD(p,g) for p,g in zip(pred,gt)] 
        for row in zip(subjects,dsc,asd,hd95,nsd):
            self.filestream_writer.writerow(['{:.2f}'.format(item) if isinstance(item, np.ndarray) or type(item) == float or type(item) == np.float64 else item for item in row])