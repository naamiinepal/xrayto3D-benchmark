from monai.transforms import *
import numpy as np
from monai.data.image_reader import PILReader
import einops

def get_lidc_transforms():
    lidc_ap_transform = Compose(
        [
            LoadImageD(
                keys={"ap"},
                ensure_channel_first=False,
                dtype=np.uint8,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=lambda image: image.rotate(90)),
            ),
            LambdaD(keys={"ap"}, func=lambda t: einops.repeat(t, "m n -> k m n", k=96)),
        ]
    )
    lidc_lat_transform = Compose(
        [
            LoadImageD(
                keys={"lat"},
                ensure_channel_first=False,
                dtype=np.uint8,
                simple_keys=True,
                image_only=False,
                reader=PILReader(converter=None),
            ),
            LambdaD(keys={"lat"}, func=lambda t: einops.repeat(t, "m n -> m n k", k=96)),
        ]
    )
    lidc_seg_transform = Compose(
        [
            LoadImageD(
                keys={"seg"},
                ensure_channel_first=True,
                dtype=np.uint8,
                simple_keys=True,
                image_only=False,
            ),
            SpacingD(
                keys={"seg"},
                pixdim=(1, 1, 1),
                mode="nearest",
                padding_mode="zeros",
            ),
            SpatialPadD(keys={"seg"}, spatial_size=(96, 96, 96)),
            OrientationD(keys={"seg"}, axcodes="PIR"),
        ]
    )
    return {'ap':lidc_ap_transform,'lat':lidc_lat_transform,'seg':lidc_seg_transform}

