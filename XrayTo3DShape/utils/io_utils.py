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
        self.nifti_saver.save_batch(prediction['pred'],prediction['seg_meta_dict'])


def parse_training_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('trainpaths')
    parser.add_argument('valpaths')
    parser.add_argument('--tags',nargs='*')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--accelerator',default='gpu')
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--res',type=float,default=1.5)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--evaluate',default=False,action='store_true')
    parser.add_argument('--save_predictions',default=False,action='store_true')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--precision',default=32,type=int)

    args = parser.parse_args()

    if args.precision == 16: args.precision = 'bf16'
    return args
