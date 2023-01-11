from monai.transforms import *
import numpy as np
from monai.data.image_reader import PILReader
import einops

def get_nonkasten_transforms(size=64,resolution=1.5):
    lidc_ap_transform = Compose(
        [
            LoadImageD(
                keys={"ap"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=lambda image: image.rotate(90)),
            ),
            EnsureChannelFirstD(keys='ap'),
            ResizeD(keys={'ap'},spatial_size=[size,size],size_mode='all',mode='bilinear'),
            # LambdaD(keys={"ap"}, func=lambda t: einops.repeat(t, "1 m n -> 1 k m n", k=size)),
            ScaleIntensityD(keys={'ap'}),
            # DataStatsD(keys='ap')
        ]
    )
    lidc_lat_transform = Compose(
        [
            LoadImageD(
                keys={"lat"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=None),
            ),
            EnsureChannelFirstD(keys='lat'),
            ResizeD(keys={'lat'},spatial_size=[size,size],size_mode='all',mode='bilinear'),            
            # LambdaD(keys={"lat"}, func=lambda t: einops.repeat(t, "1 m n -> 1 m n k", k=size)),
            ScaleIntensityD(keys={'lat'}),
            # DataStatsD(keys='lat')
        ]
    )
    lidc_seg_transform = Compose(
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
            SpatialPadD(keys={"seg"}, spatial_size=(size, size, size)),
            OrientationD(keys={"seg"}, axcodes="PIR"),
            ThresholdIntensityD(keys='seg',threshold=0.5, above=False,cval=1.0),
            # DataStatsD(keys='seg')
        ]
    )
    return {'ap':lidc_ap_transform,'lat':lidc_lat_transform,'seg':lidc_seg_transform}

def get_kasten_transforms(size=64,resolution=1.5):
    lidc_ap_transform = Compose(
        [
            LoadImageD(
                keys={"ap"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=lambda image: image.rotate(90)),
            ),
            EnsureChannelFirstD(keys='ap'),
            ResizeD(keys={'ap'},spatial_size=[size,size],size_mode='all',mode='bilinear'),
            LambdaD(keys={"ap"}, func=lambda t: einops.repeat(t, "1 m n -> 1 k m n", k=size)),
            ScaleIntensityD(keys={'ap'}),
            # DataStatsD(keys='ap')
        ]
    )
    lidc_lat_transform = Compose(
        [
            LoadImageD(
                keys={"lat"},
                ensure_channel_first=False,
                dtype=np.float32,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=None),
            ),
            EnsureChannelFirstD(keys='lat'),
            ResizeD(keys={'lat'},spatial_size=[size,size],size_mode='all',mode='bilinear'),            
            LambdaD(keys={"lat"}, func=lambda t: einops.repeat(t, "1 m n -> 1 m n k", k=size)),
            ScaleIntensityD(keys={'lat'}),
            # DataStatsD(keys='lat')
        ]
    )
    lidc_seg_transform = Compose(
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
            SpatialPadD(keys={"seg"}, spatial_size=(size, size, size)),
            OrientationD(keys={"seg"}, axcodes="PIR"),
            ThresholdIntensityD(keys='seg',threshold=0.5, above=False,cval=1.0),
            # DataStatsD(keys='seg')
        ]
    )
    return {'ap':lidc_ap_transform,'lat':lidc_lat_transform,'seg':lidc_seg_transform}

