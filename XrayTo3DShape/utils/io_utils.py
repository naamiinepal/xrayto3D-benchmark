"""utils for i/o operations"""
from pathlib import Path
import SimpleITK as sitk
import numpy as np
import json

def load_json(fp):
    with open(fp) as f:
        data = json.load(f)
    return data

def get_nifti_stem(path) -> str:
    """
    '/home/user/image.nii.gz' -> 'image'
    1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235.nii.gz
    ->
    1.3.6.1.4.1.14519.5.2.1.6279.6001.905371958588660410240398317235
    """

    def _get_stem(path_string) -> str:
        name_subparts = Path(path_string).name.split(".")
        return ".".join(name_subparts[:-2])  # get rid of nii.gz

    return _get_stem(path)


def to_numpy(x) -> np.ndarray:
    """detach and convert torch tensor to numpy"""
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return x


def write_image(img, out_path, pixeltype=None):
    """wrap simpleitk WriteImage"""
    if isinstance(out_path, Path):
        out_path = str(out_path)
    if pixeltype:
        img = sitk.Cast(img, pixeltype)
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
