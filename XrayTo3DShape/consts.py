from .architectures import CustomAutoEncoder,TLPredictor,MultiScale2DPermuteConcat,OneDConcat,TwoDPermuteConcat
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unet import UNet

from .experiments import ParallelHeadsExperiment,VolumeAsInputExperiment,TLPredictorExperiment,AutoencoderExperiment

from .transforms import get_kasten_transforms,get_nonkasten_transforms, get_denoising_autoencoder_transforms

from monai.transforms.compose import Compose
from typing import Dict

anatomy_resolution_dict  = {'totalseg_femur':(128,1.0),
                       'totalseg_ribs':(320,1.0),
                       'totalseg_hips':(288,1.0),
                       'verse2019':(96,1.0),
                       'verse2020':(96,1.0),
                       'femur':(128,1.0),
                       'rib':(320,1.0),
                       'hip':(288,1.0),
                       'verse':(96,1.0),
                       }

model_experiment_dict = {
    CustomAutoEncoder.__name__ : AutoencoderExperiment.__name__,
    TLPredictor.__name__ : TLPredictorExperiment.__name__,
    UNet.__name__ : VolumeAsInputExperiment.__name__,
    AttentionUnet.__name__: VolumeAsInputExperiment.__name__,
    TwoDPermuteConcat.__name__ : ParallelHeadsExperiment.__name__,
    OneDConcat.__name__: ParallelHeadsExperiment.__name__,
    MultiScale2DPermuteConcat.__name__ : ParallelHeadsExperiment.__name__,
}

def get_transform_from_model_name(model_name,image_size,resolution)->Dict[str,Compose]:
    experiment_name = model_experiment_dict[model_name]
    if experiment_name == ParallelHeadsExperiment.__name__:
        callable_transform = get_nonkasten_transforms(size=image_size,resolution=resolution)
    elif experiment_name == VolumeAsInputExperiment.__name__ or experiment_name == TLPredictorExperiment.__name__:
        callable_transform = get_kasten_transforms(size=image_size,resolution=resolution)
    elif experiment_name == AutoencoderExperiment.__name__:
        callable_transform = get_denoising_autoencoder_transforms(size=image_size,resolution=resolution)
    else:
        raise ValueError(f'Invalid experiment name {experiment_name}')
    return callable_transform
