# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""utils for numpy array"""
from pathlib import Path
from typing import List, Union

import numpy as np
import SimpleITK as sitk
import torch

from .io_utils import write_image


def repeat_along_dim(img: np.ndarray, dim: int, times: int):
    """make 2D slice into 3D volume by repeating along given dimensions"""
    if dim == 0:
        return np.repeat(img[np.newaxis, :, :], times, axis=dim)
    if dim == 1:
        return np.repeat(img[:, np.newaxis, :], times, axis=dim)
    if dim == 2:
        return np.repeat(img[:, :, np.newaxis], times, axis=dim)
    raise ValueError(f"dimension {dim} should be in the range [0-2]")


def reproject(volume: Union[torch.Tensor, np.ndarray], dim: int):
    """mean projection of volume"""
    if isinstance(volume, torch.Tensor):
        return torch.mean(volume, dim=dim) * 255
    if isinstance(volume, np.ndarray):
        return np.mean(volume, axis=dim) * 255


def get_projectionslices_from_3d(image) -> List:
    """project 3D volumes along all three dimension"""
    assert len(image.shape) == 3, f'image should be 3-dim but got {len(image.shape)}-dim'
    return [reproject(image, dim=i) for i in range(3)]


def save_numpy_as_nifti(volume_np: np.ndarray, out_path: Union[str, Path]):
    """save numpy array as nifti volume"""
    volume_sitk = sitk.GetImageFromArray(np.asarray(volume_np, dtype=np.uint8))
    write_image(volume_sitk, out_path)
