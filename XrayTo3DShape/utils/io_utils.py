from pathlib import Path
import SimpleITK as sitk
from monai.data.nifti_saver import NiftiSaver
from pytorch_lightning.callbacks import BasePredictionWriter
import pytorch_lightning as pl
from typing import Any,Sequence,Optional
import numpy as np

def write_image(img, out_path,pixeltype=None):
    if isinstance(out_path, Path):
        out_path = str(out_path)
    if pixeltype:
        img = sitk.Cast(img,pixeltype)
    sitk.WriteImage(img, out_path)

def read_image(img_path):
    """returns the SimpleITK image read from given path

    Parameters:
    -----------
    pixeltype (ImagePixelType):
    """

    if isinstance(img_path, Path):
        img_path = str(img_path)

    # if pixeltype == ImagePixelType.ImageType:
    #     pixeltype = sitk.sitkUInt16
    #     return sitk.ReadImage(img_path,pixeltype)

    # elif pixeltype == ImagePixelType.SegmentationType:
    #     pixeltype = sitk.sitkUInt8
    #     return sitk.ReadImage(img_path,pixeltype)

    # else:
    #     raise ValueError(f'ImagePixelType cannot be {pixeltype}')

    return sitk.ReadImage(img_path)


class NiftiPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval: str = "batch") -> None:
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.nifti_saver = NiftiSaver(output_dir=self.output_dir,output_postfix='pred',resample=True,dtype=np.int16,separate_folder=False)
    
    def write_on_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", prediction: Any, batch_indices: Optional[Sequence[int]], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self.nifti_saver.save_batch(prediction['pred_seg'],prediction['pred_seg_meta_dict'])

