"""specialized data transformation specific to model architecture"""
from typing import Sequence

import einops
import numpy as np
import torch
from monai.data.image_reader import PILReader

from monai.transforms.spatial.dictionary import ResizeD, SpacingD, OrientationD
from monai.transforms.intensity.dictionary import ThresholdIntensityD, ScaleIntensityD
from monai.transforms.io.dictionary import LoadImageD
from monai.transforms.utility.dictionary import EnsureChannelFirstD, LambdaD
from monai.transforms.utility.array import Lambda
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropD
from monai.transforms.compose import Compose
from skimage.util import random_noise


def get_resize_transform(
    keys, original_size: Sequence[int], reduction_rate: float, mode="nearest"
):
    """thin Wrapper around ```monai.transforms.ResizeD``` to resize an image by given reduction rate

    Args:
        keys (_type_): Same as monai keys passed to callable transform
        original_size (Sequence): _description_
        reduction_rate (float): _description_
        mode (str, optional): _description_. Defaults to "nearest".

    Returns:
        monai.transforms: callable transform
    """
    target_size = list(map(lambda x: x / reduction_rate, original_size))
    return ResizeD(keys=keys, spatial_size=target_size, mode=mode, align_corners=True)


def get_denoising_autoencoder_transforms(size=64, resolution=1.5):
    """return both noisy and clean version of the segmentation label

    Args:
        size (int, optional): image size. Defaults to 64.
        resolution (float, optional): image resolution. Defaults to 1.5.

    Returns:
        dict[str,monai.transforms]: ap,lat,seg callable transforms
    """
    noise_lambda = Lambda(
        lambda d: {
            "orig": d["seg"],
            "gaus": torch.tensor(
                random_noise(d["seg"], mode="gaussian", mean=0, var=0.01, clip=False),
                dtype=torch.float32,
            ),
        }
    )

    seg_transform = Compose(
        [
            LoadImageD(
                keys={"seg"},
                ensure_channel_first=True,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
            ),
            SpacingD(
                keys={"seg"},
                pixdim=(resolution, resolution, resolution),
                mode="nearest",
                padding_mode="zeros",
            ),
            ResizeWithPadOrCropD(keys={"seg"}, spatial_size=(size, size, size)),
            OrientationD(keys={"seg"}, axcodes="PIR"),
            ThresholdIntensityD(keys="seg", threshold=0.5, above=False, cval=1.0),
            noise_lambda
            # DataStatsD(keys='seg')
        ]
    )

    def identity(x):
        return x

    return {
        "ap": identity,
        "lat": identity,
        "seg": seg_transform,
    }


def get_nonkasten_transforms(size=64, resolution=1.5):
    """transform AP and LAT images as 2D images"""
    ap_transform = Compose(
        [
            LoadImageD(
                keys={"ap"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=lambda image: image.rotate(90)),
            ),
            EnsureChannelFirstD(keys="ap"),
            ResizeD(
                keys={"ap"},
                spatial_size=[size, size],
                size_mode="all",
                mode="bilinear",
                align_corners=True,
            ),
            # LambdaD(keys={"ap"}, func=lambda t: einops.repeat(t, "1 m n -> 1 k m n", k=size)),
            ScaleIntensityD(keys={"ap"}),
            # DataStatsD(keys='ap')
        ]
    )
    lat_transform = Compose(
        [
            LoadImageD(
                keys={"lat"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=None),
            ),
            EnsureChannelFirstD(keys="lat"),
            ResizeD(
                keys={"lat"},
                spatial_size=[size, size],
                size_mode="all",
                mode="bilinear",
                align_corners=True,
            ),
            # LambdaD(keys={"lat"}, func=lambda t: einops.repeat(t, "1 m n -> 1 m n k", k=size)),
            ScaleIntensityD(keys={"lat"}),
            # DataStatsD(keys='lat')
        ]
    )
    seg_transform = Compose(
        [
            LoadImageD(
                keys={"seg"},
                ensure_channel_first=True,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
            ),
            SpacingD(
                keys={"seg"},
                pixdim=(resolution, resolution, resolution),
                mode="nearest",
                padding_mode="zeros",
            ),
            ResizeWithPadOrCropD(keys={"seg"}, spatial_size=(size, size, size)),
            OrientationD(keys={"seg"}, axcodes="PIR"),
            ThresholdIntensityD(keys="seg", threshold=0.5, above=False, cval=1.0),
            # DataStatsD(keys='seg')
        ]
    )
    return {"ap": ap_transform, "lat": lat_transform, "seg": seg_transform}


def get_kasten_transforms(size=64, resolution=1.5):
    """transform AP/LAT images into 3D volumes by repeating along orthogonal directions"""
    ap_transform = Compose(
        [
            LoadImageD(
                keys={"ap"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=lambda image: image.rotate(90)),
            ),
            EnsureChannelFirstD(keys="ap"),
            ResizeD(
                keys={"ap"},
                spatial_size=[size, size],
                size_mode="all",
                mode="bilinear",
                align_corners=True,
            ),
            LambdaD(
                keys={"ap"}, func=lambda t: einops.repeat(t, "1 m n -> 1 k m n", k=size)
            ),
            ScaleIntensityD(keys={"ap"}),
            # DataStatsD(keys='ap')
        ]
    )
    lat_transform = Compose(
        [
            LoadImageD(
                keys={"lat"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=None),
            ),
            EnsureChannelFirstD(keys="lat"),
            ResizeD(
                keys={"lat"},
                spatial_size=[size, size],
                size_mode="all",
                mode="bilinear",
                align_corners=True,
            ),
            LambdaD(
                keys={"lat"},
                func=lambda t: einops.repeat(t, "1 m n -> 1 m n k", k=size),
            ),
            ScaleIntensityD(keys={"lat"}),
            # DataStatsD(keys='lat')
        ]
    )
    seg_transform = Compose(
        [
            LoadImageD(
                keys={"seg"},
                ensure_channel_first=True,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
            ),
            OrientationD(keys={"seg"}, axcodes="PIR"),
            SpacingD(
                keys={"seg"},
                pixdim=(resolution, resolution, resolution),
                mode="nearest",
                padding_mode="zeros",
            ),
            ResizeWithPadOrCropD(keys={"seg"}, spatial_size=(size, size, size)),
            ThresholdIntensityD(keys="seg", threshold=0.5, above=False, cval=1.0),
            # DataStatsD(keys='seg')
        ]
    )
    return {"ap": ap_transform, "lat": lat_transform, "seg": seg_transform}
