from typing import List, Union
import numpy as np
import torch
import SimpleITK as sitk
from .io_utils import write_image

def repeat_along_dim(img: np.ndarray, dim, times):
    if dim == 0:
        return np.repeat(img[np.newaxis, :, :], times, axis=dim)
    if dim == 1:
        return np.repeat(img[:, np.newaxis, :], times, axis=dim)
    if dim == 2:
        return np.repeat(img[:, :, np.newaxis], times, axis=dim)
    raise ValueError(f'dimension {dim} should be in the range [0-2]')

def reproject(volume: Union[torch.Tensor, np.ndarray], dim):
    if isinstance(volume, torch.Tensor):
        return torch.sum(volume, dim=dim)
    if isinstance(volume, np.ndarray):
        return np.sum(volume, axis=dim)


def get_projectionslices_from_3d(image) -> List:
    return [reproject(image, dim=i) for i in range(3)]


def save_numpy_as_nifti(volume_np: np.ndarray, out_path):
    volume_sitk = sitk.GetImageFromArray(np.asarray(volume_np, dtype=np.uint8))
    write_image(volume_sitk, out_path)
