# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""Work in Progress: Data transformation pipeline for Atlas-deformation based architecture"""
from monai.transforms.spatial.dictionary import ResizeD, SpacingD, OrientationD
from monai.transforms.intensity.dictionary import ThresholdIntensityD, ScaleIntensityD
from monai.transforms.io.dictionary import LoadImageD
from monai.transforms.utility.dictionary import EnsureChannelFirstD
from monai.transforms.croppad.dictionary import ResizeWithPadOrCropD
from monai.transforms.compose import Compose
from monai.data.image_reader import PILReader
import numpy as np


def get_atlas_deformation_transforms(size=64, resolution=1.5):
    """transformation pipeline for Atlas-deformation based model architecture"""
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
                align_corners=True
            ),
            ResizeWithPadOrCropD(keys={"seg"}, spatial_size=(size, size, size)),
            OrientationD(keys={"seg"}, axcodes="PIR"),
            ThresholdIntensityD(keys="seg", threshold=0.5, above=False, cval=1.0),
            # DataStatsD(keys='seg')
        ]
    )

    atlas_transform = Compose(
        [
            LoadImageD(
                keys={"atlas"},
                ensure_channel_first=True,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
            ),
            SpacingD(
                keys={"atlas"},
                pixdim=(resolution, resolution, resolution),
                mode="nearest",
                padding_mode="zeros",
                align_corners=True
            ),
            ResizeWithPadOrCropD(keys={"atlas"}, spatial_size=(size, size, size)),
            OrientationD(keys={"atlas"}, axcodes="PIR"),
            ThresholdIntensityD(keys="atlas", threshold=0.5, above=False, cval=1.0),
            # DataStatsD(keys='seg')
        ]
    )
    return {
        "ap": ap_transform,
        "lat": lat_transform,
        "seg": seg_transform,
        "atlas": atlas_transform,
    }


def get_deformation_transforms(size=64, resolution=1.5):
    """transformation pipeline for image registration.
    Fixed image and moving image transforms.
    """
    fixed_transform = Compose(
        [
            LoadImageD(
                keys="fixed",
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(),
            ),
            EnsureChannelFirstD(keys="fixed"),
            ResizeD(
                keys="fixed",
                spatial_size=[size, size],
                size_mode="all",
                mode="bilinear",
                align_corners=True,
            ),
            ScaleIntensityD(keys="fixed"),
        ]
    )
    moving_transform = Compose(
        [
            LoadImageD(
                keys="moving",
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(),
            ),
            EnsureChannelFirstD(keys="moving"),
            ResizeD(
                keys="moving",
                spatial_size=[size, size],
                size_mode="all",
                mode="bilinear",
                align_corners=True,
            ),
            ScaleIntensityD(keys="moving"),
        ]
    )
    return {"fixed": fixed_transform, "moving": moving_transform}
