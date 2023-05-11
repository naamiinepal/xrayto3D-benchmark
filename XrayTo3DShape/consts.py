"""
Contains Image size of datasets from the preprocessing pipeline.
returns model architecture-specific data transformation pipeline
"""
from typing import Dict

from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unet import UNet
from monai.networks.nets.unetr import UNETR
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.transforms.compose import Compose

from .architectures import (
    CustomAutoEncoder,
    MultiScale2DPermuteConcat,
    OneDConcat,
    TLPredictor,
    TwoDPermuteConcat,
)
from .experiments import (
    AutoencoderExperiment,
    ParallelHeadsExperiment,
    TLPredictorExperiment,
    VolumeAsInputExperiment,
)
from .transforms import (
    get_denoising_autoencoder_transforms,
    get_kasten_transforms,
    get_nonkasten_transforms,
)

# these dataset were originally defined for (size,resolution)
anatomy_resolution_dict = {
    "totalseg_femur": (128, 1.0),
    "totalseg_ribs": (320, 1.0),
    "totalseg_hips": (288, 1.0),
    "verse2019": (96, 1.0),
    "verse2020": (96, 1.0),
    "vertebra": (96, 1.0),
    "femur": (128, 1.0),
    "rib": (320, 1.0),
    "hip": (288, 1.0),
    "verse": (96, 1.0),
}
MODEL_NAMES = [
    "SwinUNETR",
    "UNETR",
    "AttentionUnet",
    "UNet",
    "MultiScale2DPermuteConcat",
    "TwoDPermuteConcat",
    "OneDConcat",
    "TLPredictor",
]
# model architecture : architecture dependent training regime
model_experiment_dict = {
    CustomAutoEncoder.__name__: AutoencoderExperiment.__name__,
    TLPredictor.__name__: TLPredictorExperiment.__name__,
    UNet.__name__: VolumeAsInputExperiment.__name__,
    AttentionUnet.__name__: VolumeAsInputExperiment.__name__,
    UNETR.__name__: VolumeAsInputExperiment.__name__,
    SwinUNETR.__name__: VolumeAsInputExperiment.__name__,
    TwoDPermuteConcat.__name__: ParallelHeadsExperiment.__name__,
    OneDConcat.__name__: ParallelHeadsExperiment.__name__,
    MultiScale2DPermuteConcat.__name__: ParallelHeadsExperiment.__name__,
}


def get_transform_from_model_name(
    model_name, image_size, resolution
) -> Dict[str, Compose]:
    """return data transformation pipeline for given
    model architecture and suggested target size and resolution

    Args:
        model_name (str): model architecture name
        image_size (int): target dimension of images
        resolution (float): (isotropic) voxel resolution

    Raises:
        ValueError: If model name does not match

    Returns:
        Dict[str, Compose]: Callable Transform
    """
    experiment_name = model_experiment_dict[model_name]
    if experiment_name == ParallelHeadsExperiment.__name__:
        callable_transform = get_nonkasten_transforms(
            size=image_size, resolution=resolution
        )
    elif experiment_name in (
        VolumeAsInputExperiment.__name__,
        TLPredictorExperiment.__name__,
    ):
        callable_transform = get_kasten_transforms(
            size=image_size, resolution=resolution
        )
    elif experiment_name == AutoencoderExperiment.__name__:
        callable_transform = get_denoising_autoencoder_transforms(
            size=image_size, resolution=resolution
        )
    else:
        raise ValueError(f"Invalid experiment name {experiment_name}")
    return callable_transform
