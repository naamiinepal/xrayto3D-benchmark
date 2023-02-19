from .oneDConcat_model import OneDConcatModel
from .twoDPermuteConcat_model import TwoDPermuteConcatModel
from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.unet import Unet
from .utils import calculate_1d_vec_channels
from torch import nn

def get_model(model_name,image_size)->nn.Module:
    if model_name == OneDConcatModel.__name__:
        return OneDConcatModel(get_1dconcatmodel_config(image_size))
    elif model_name == AttentionUnet.__name__:
        return AttentionUnet(spatial_dims=3,**get_attunet_config())
    elif model_name == TwoDPermuteConcatModel.__name__:
        return TwoDPermuteConcatModel(get_2dconcatmodel_config(image_size))
    elif model_name == Unet.__name__:
        return Unet(spatial_dims=3,**get_unet_config())
    else:
        raise ValueError(f'invalid model name {model_name}')

def get_model_config(model_name,image_size):
    if model_name == OneDConcatModel.__name__:
        return get_1dconcatmodel_config(image_size)
    elif model_name == AttentionUnet.__name__:
        return get_attunet_config()
    elif model_name == TwoDPermuteConcatModel.__name__:
        return get_2dconcatmodel_config(image_size)
    elif model_name == Unet.__name__:
        return get_unet_config()
    else:
        raise ValueError(f'invalud model name {model_name}')

def get_unet_config():
    # End-To-End Convolutional Neural Network for 3D Reconstruction of Knee Bones From Bi-Planar X-Ray Images
    # https://arxiv.org/pdf/2004.00871.pdf
    model_config = {
        "in_channels": 2,
        "out_channels": 1,
        "channels": (8, 16, 32, 64, 128),
        "strides": (2,2,2,2),
        "act": "RELU",
        "norm": "BATCH",
        "num_res_units": 2,
        "dropout": 0.1,
    }
    return model_config

def get_2dconcatmodel_config(image_size):
    # Inferring the 3D Standing Spine Posture from 2D Radiographs
    # https://arxiv.org/abs/2007.06612
    # default baseline for 64^3 volume
    if (image_size == 128) or (image_size == 64):
        model_config = {
            "input_image_size": [image_size, image_size],
            "encoder": {
                "in_channels": [1, 16, 32, 32, 32, 32],
                "out_channels": [16, 32, 32, 32, 32, 32],
                "strides": [2, 2, 1, 1, 1, 1],
                "kernel_size": 7,
            },
            "ap_expansion": {
                "in_channels": [32, 32, 32, 32],
                "out_channels": [32, 32, 32, 32],
                "strides": ((2, 1, 1),) * 4,
                "kernel_size": 3,
            },
            "lat_expansion": {
                "in_channels": [32, 32, 32, 32],
                "out_channels": [32, 32, 32, 32],
                "strides": ((1, 1, 2),) * 4,
                "kernel_size": 3,
            },
            "decoder": {
                "in_channels": [64, 64, 64, 64, 64, 32, 16],
                "out_channels": [64, 64, 64, 64, 32, 16, 1],
                "strides": (1, 1, 1, 1, 2, 2, 1),
                "kernel_size": (3,3,3,3,3,3,7),
            },
            "act": "RELU",
            "norm": "BATCH",
            "dropout": 0.0,
            "bias": False
        }
        if image_size == 128:
            # add a decoder layer
            model_config['decoder']["in_channels"] = [64, 64, 64, 64, 64, 32, 16]
            model_config['decoder']["out_channels"] = [64, 64, 64, 64, 32, 16, 1]
            model_config['decoder']["strides"] = (1, 1, 1, 1, 2, 2, 1)
            model_config['decoder']["kernel_size"] = (3,3,3,3,3,3,7)
    else:
        raise ValueError(f'Image size can be either 64 or 128,, got {image_size}')

    return model_config

def get_attunet_config():
    # Attention U-Net: Learning Where to Look for the Pancreas
    # https://arxiv.org/abs/1804.03999 
    model_config = {
        "in_channels": 2,
        "out_channels": 1,
        "channels": (8, 16, 32, 64, 64),
        "strides": (2,2,2,2),
    }
    return model_config

def get_1dconcatmodel_config(image_size):
    # base model for 128^3 volume (2^7)
    if (image_size == 128) or (image_size == 64) :
        model_config = {
            "input_image_size": [image_size, image_size],
            "encoder": {
                "in_channels": [1, 32, 64, 128, 256],
                "out_channels": [32, 64, 128, 256, 256],
                "strides": [2, 2, 2, 2, 2],
            },
            "decoder": {
                "in_channels": [1024, 512, 8, 4, 4, 4],
                "out_channels": [1024, 512, 8, 4, 4, 4, 1],
                "strides": [2,]*7,
            },
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
            "dropout": 0.0,
            "bias": True,
        }
        if image_size == 64:
            # remove a decoder layer for 64^3 voume
            model_config['decoder']['in_channels'] = [1024, 512, 8, 4, 4]
            model_config['decoder']['out_channels'] = [1024, 512, 8, 4, 4, 1]
            model_config['decoder']['strides'] = [2,]*6
    else:
        raise ValueError(f'Image size can be either 64 or 128,, got {image_size}')

    embedding_vec_size = calculate_1d_vec_channels(image_size,model_config['encoder']['strides'],model_config['encoder']['out_channels'][-1])
    model_config['decoder']['in_channels'].insert(0,embedding_vec_size)
    return model_config