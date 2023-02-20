from pathlib import Path
import SimpleITK as sitk
import numpy as np
import os

def get_nifti_stem(path)->str:
    """
    '/home/user/image.nii.gz' -> 'image'
    1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235.nii.gz ->1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235
    """
    def _get_stem(path_string) -> str:
        name_subparts = Path(path_string).name.split('.')
        return '.'.join(name_subparts[:-2]) # get rid of nii.gz
    return _get_stem(path)



def to_numpy(x)->np.ndarray:
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x

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




def parse_training_arguments():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('trainpaths')
    parser.add_argument('valpaths')
    parser.add_argument('--model_name')
    parser.add_argument('--experiment_name')
    parser.add_argument('--anatomy')
    parser.add_argument('--loss')
    parser.add_argument('--tags',nargs='*')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--accelerator',default='gpu')
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--res',type=float,default=1.5)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--dropout',default=False,type=bool)
    parser.add_argument('--evaluate',default=False,action='store_true')
    parser.add_argument('--save_predictions',default=False,action='store_true')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--precision',default=32,type=int)

    args = parser.parse_args()

    if args.precision == 16: args.precision = 'bf16'
    return args
