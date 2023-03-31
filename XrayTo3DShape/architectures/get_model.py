import math
from typing import Dict

from monai.networks.nets.attentionunet import AttentionUnet
from monai.networks.nets.autoencoder import AutoEncoder
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.unet import Unet
from monai.networks.nets.unetr import UNETR
from torch import nn

from .autoencoder_v2 import CustomAutoEncoder, TLPredictor
from .oneDConcat_model import OneDConcat
from .twoDPermuteConcat_model import TwoDPermuteConcat
from .twoDPermuteConcatMultiScale import MultiScale2DPermuteConcat


def get_model(model_name, image_size, dropout=False) -> nn.Module:
    if model_name in (OneDConcat.__name__, "OneDConcatModel"):
        return OneDConcat(get_1dconcatmodel_config(image_size))

    elif model_name == AttentionUnet.__name__:
        return AttentionUnet(spatial_dims=3, **get_attunet_config())

    elif model_name == SwinUNETR.__name__:
        return SwinUNETR(**get_swinunetr_config(image_size))

    elif model_name == UNETR.__name__:
        return UNETR(**get_unetr_config(image_size, dropout=dropout))

    elif model_name in (TwoDPermuteConcat.__name__, "TwoDPermuteConcatModel"):
        return TwoDPermuteConcat(get_2dconcatmodel_config(image_size))

    elif model_name == Unet.__name__:
        return Unet(spatial_dims=3, **get_unet_config(dropout))

    elif model_name == MultiScale2DPermuteConcat.__name__:
        return MultiScale2DPermuteConcat(get_multiscale2dconcatmodel_config(image_size))

    elif model_name == CustomAutoEncoder.__name__:
        return CustomAutoEncoder(**get_autoencoder_config(image_size))

    elif model_name == TLPredictor.__name__:
        return TLPredictor(**get_tlpredictor_config(image_size))
    else:
        raise ValueError(f"invalid model name {model_name}")


def get_model_config(model_name, image_size, dropout=False):
    if model_name in (OneDConcat.__name__, "OneDConcatModel"):
        return get_1dconcatmodel_config(image_size)
    elif model_name == AttentionUnet.__name__:
        return get_attunet_config()
    elif model_name == SwinUNETR.__name__:
        return get_swinunetr_config(image_size)
    elif model_name == UNETR.__name__:
        return get_unetr_config(image_size, dropout)
    elif model_name in (TwoDPermuteConcat.__name__, "TwoDPermuteConcatModel"):
        return get_2dconcatmodel_config(image_size)
    elif model_name == Unet.__name__:
        return get_unet_config(dropout)
    elif model_name == MultiScale2DPermuteConcat.__name__:
        return get_multiscale2dconcatmodel_config(image_size)
    elif model_name == AutoEncoder.__name__:
        return get_autoencoder_config(image_size)
    elif model_name == CustomAutoEncoder.__name__:
        return get_autoencoder_config(image_size)
    elif model_name == TLPredictor.__name__:
        return get_tlpredictor_config(image_size)
    else:
        raise ValueError(f"invalid model name {model_name}")


def get_unetr_config(image_size, dropout):
    model_config = {
        "in_channels": 2,
        "out_channels": 1,
        "img_size": image_size,
        "dropout_rate": 0.1 if dropout else 0.0,
    }
    return model_config


def get_tlpredictor_config(image_size):
    model_config = {
        "image_size": image_size,
        "spatial_dims": 3,
        "in_channel": 2,
        "latent_dim": 64,
        "kernel_size": 3,
        "channels": [8, 16, 32, 64],
        "strides": [2, 2, 2, 2],
    }
    return model_config


def get_autoencoder_config(image_size):
    model_config = {
        "image_size": image_size,
        "latent_dim": 64,
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 1,
        "channels": (8, 16, 32, 64),
        "strides": (2, 2, 2, 2),
    }
    return model_config


def get_swinunetr_config(image_size) -> Dict:
    model_config = {
        "img_size": image_size,
        "in_channels": 2,
        "out_channels": 1,
        "depths": (2, 2, 2, 2),
        "num_heads": (2, 2, 2, 2),
        "feature_size": 12,
        "norm_name": "instance",
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "dropout_path_rate": 0.0,
        "normalize": True,
        "use_checkpoint": True,
        "spatial_dims": 3,
        "downsample": "merging",
    }
    return model_config


def get_unet_config(dropout):
    """End-To-End Convolutional Neural Network for 3D Reconstruction of Knee Bones From Bi-Planar X-Ray Images
    https://arxiv.org/pdf/2004.00871.pdf
    """
    model_config = {
        "in_channels": 2,
        "out_channels": 1,
        "channels": (8, 16, 32, 64, 128),
        "strides": (2, 2, 2, 2),
        "act": "RELU",
        "norm": "BATCH",
        "num_res_units": 2,
        "dropout": 0.1 if dropout else 0.0,
    }
    return model_config


def get_multiscale2dconcatmodel_config(image_size):
    """fully conv: image size does not matter"""
    model_config = {
        "permute": True,
        "encoder": {
            "initial_channel": 16,
            "in_channels": [],  # this will be filled in by autoconfig
            "out_channels": [4, 8, 16, 32, 64],
            "encoder_count": 4,
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
        },
        "decoder_2D": {
            "in_channels": [],  # this will be filled in by autoconfig
            "out_channels": [8, 16, 32, 64, 128]
            if image_size == 128
            else [4, 8, 16, 32, 64],
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
        },
        "fusion_3D": {
            "in_channels": [],  # this will be filled in by autoconfig
            "out_channels": [32, 32, 32, 32, 32],
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
        },
    }
    # constrain decoder configuration based on encoder
    enc_in = []
    for i in range(len(model_config["encoder"]["out_channels"])):
        if i == 0:
            enc_in.append(model_config["encoder"]["initial_channel"])
        else:
            enc_in.append(
                model_config["encoder"]["out_channels"][i - 1]
                * model_config["encoder"]["encoder_count"]
                + enc_in[i - 1]
            )
    model_config["encoder"]["in_channels"] = enc_in

    enc_out = [
        in_ch + out_ch * model_config["encoder"]["encoder_count"]
        for in_ch, out_ch in zip(
            model_config["encoder"]["in_channels"],
            model_config["encoder"]["out_channels"],
        )
    ]
    enc_out.reverse()
    dec_out = model_config["decoder_2D"]["out_channels"]
    dec_in = []
    for i in range(len(enc_out)):
        # decoder takes input from the adjacent encoder and previous decoder
        if i == 0:  # if this is the first decoder, then there is no previous decoder
            dec_in.append(enc_out[i])
        else:
            dec_in.append(enc_out[i] + dec_out[i - 1])
    model_config["decoder_2D"]["in_channels"] = dec_in

    fusion_in = []
    fusion_out = model_config["fusion_3D"]["out_channels"]
    for i in range(len(fusion_out)):
        if i == 0:
            fusion_in.append(2)  # AP and LAT view reshaped into 3D view
        else:
            fusion_in.append(
                2 + fusion_out[i - 1]
            )  # 2 channels from AP and LAT decoder, additional channel from earlier fusion decoder
    model_config["fusion_3D"]["in_channels"] = fusion_in

    return model_config


def get_2dconcatmodel_config(image_size):
    """Inferring the 3D Standing Spine Posture from 2D Radiographs
    https://arxiv.org/abs/2007.06612
    default baseline for 64^3 volume
    """
    if image_size == 64 or image_size == 128:
        expansion_depth = int(math.log2(image_size)) - 2

        model_config = {
            "input_image_size": [image_size, image_size],
            "encoder": {
                "in_channels": [1, 16, 32, 32, 32, 32],
                "out_channels": [16, 32, 32, 32, 32, 32],
                "strides": [2, 2, 1, 1, 1, 1],
                "kernel_size": 7,
            },
            "ap_expansion": {
                "in_channels": [32, 32, 32, 32, 32][:expansion_depth],
                "out_channels": [32, 32, 32, 32, 32][:expansion_depth],
                "strides": ((2, 1, 1),) * expansion_depth,
                "kernel_size": 3,
            },
            "lat_expansion": {
                "in_channels": [32, 32, 32, 32, 32][:expansion_depth],
                "out_channels": [32, 32, 32, 32, 32][:expansion_depth],
                "strides": ((1, 1, 2),) * expansion_depth,
                "kernel_size": 3,
            },
            "decoder": {
                "in_channels": [64, 64, 64, 64, 64, 32, 16],
                "out_channels": [64, 64, 64, 64, 32, 16, 1],
                "strides": (1, 1, 1, 1, 2, 2, 1),
                "kernel_size": (3, 3, 3, 3, 3, 3, 7),
            },
            "act": "RELU",
            "norm": "BATCH",
            "dropout": 0.0,
            "bias": True,
        }
    else:
        raise ValueError(f"Image size can be either 64 or 128,, got {image_size}")

    return model_config


def get_attunet_config():
    """Attention U-Net: Learning Where to Look for the Pancreas
    https://arxiv.org/abs/1804.03999
    """
    model_config = {
        "in_channels": 2,
        "out_channels": 1,
        "channels": (8, 16, 32, 64, 128),
        "strides": (2, 2, 2, 2),
    }
    return model_config


def get_1dconcatmodel_config(image_size):
    """base model for 128^3 volume (2^7)"""
    bottleneck_size = 256

    if (image_size == 128) or (image_size == 64):
        model_config = {
            "input_image_size": [image_size, image_size],
            "encoder": {
                "in_channels": [
                    1,
                    32,
                    64,
                    128,
                    256,
                    512,
                ],
                "out_channels": [32, 64, 128, 256, 512, 1024],
                "strides": [2, 2, 2, 2, 2, 2],
            },
            "decoder": {
                "in_channels": [1024, 512, 256, 128, 64, 32],
                "out_channels": [1024, 512, 256, 128, 64, 32, 1],
                "strides": [
                    2,
                ]
                * 7,
            },
            "kernel_size": 3,
            "act": "RELU",
            "norm": "BATCH",
            "dropout": 0.0,
            "bias": True,
            "bottleneck_size": bottleneck_size,
        }
        if image_size == 64:
            # remove a decoder layer for 64^3 voume
            model_config["decoder"]["in_channels"] = [
                1024,
                512,
                256,
                128,
                64,
            ]
            model_config["decoder"]["out_channels"] = [1024, 512, 256, 128, 64, 1]
            model_config["decoder"]["strides"] = [
                2,
            ] * 6
    else:
        raise ValueError(f"Image size can be either 64 or 128,, got {image_size}")

    model_config["decoder"]["in_channels"].insert(0, bottleneck_size)
    return model_config
