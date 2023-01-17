from monai.transforms import *
import numpy as np
from monai.data.image_reader import PILReader
import einops
import torch
from typing import Sequence
from skimage.util import random_noise

def get_resize_transform(keys,original_size:Sequence,reduction_rate:float,mode='nearest'):
    target_size = list(map(lambda x: x / reduction_rate, original_size)) 
    return ResizeD(keys=keys,spatial_size=target_size,mode=mode,align_corners=True)
    
def get_denoising_autoencoder_transforms(size=64, resolution=1.5):

    NoiseLambda = Lambda(lambda d: {
        "orig": d["seg"],
        "gaus": torch.tensor(
            random_noise(d["seg"], mode='gaussian',mean=0,var=0.01), dtype=torch.float32),
        "s&p": torch.tensor(random_noise(d["seg"], mode='s&p', salt_vs_pepper=0.1)),
    })

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
            NoiseLambda
            # DataStatsD(keys='seg')
        ]
    )
    callable_identity_transform = lambda x: x
    return {'ap':callable_identity_transform, 'lat':callable_identity_transform, 'seg':seg_transform}


    
def get_nonkasten_transforms(size=64, resolution=1.5):
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
            ResizeD(keys={"ap"}, spatial_size=[size, size], size_mode="all", mode="bilinear",align_corners=True),
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
            ResizeD(keys={"lat"}, spatial_size=[size, size], size_mode="all", mode="bilinear",align_corners=True),
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
            ResizeD(keys={"ap"}, spatial_size=[size, size], size_mode="all", mode="bilinear",align_corners=True),
            LambdaD(keys={"ap"}, func=lambda t: einops.repeat(t, "1 m n -> 1 k m n", k=size)),
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
            ResizeD(keys={"lat"}, spatial_size=[size, size], size_mode="all", mode="bilinear",align_corners=True),
            LambdaD(keys={"lat"}, func=lambda t: einops.repeat(t, "1 m n -> 1 m n k", k=size)),
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
