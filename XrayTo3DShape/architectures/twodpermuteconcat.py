# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


from typing import Dict, List

import torch
from monai.networks.blocks.convolutions import Convolution
from torch import nn
from ..utils.registry import ARCHITECTURES


@ARCHITECTURES.register("TwoDPermuteConcatModel")
@ARCHITECTURES.register("TwoDPermuteConcat")
class TwoDPermuteConcat(nn.Module):
    """
    Transvert Architecture
    Bayat, Amirhossein, et al. "Inferring the 3D standing spine posture from 2D radiographs.", 2020.

        Attributes:
        pa_encoder (nn.Module): encodes AP view x-rays
        lat_encoder (nn.Module): encodes LAT view x-rays
        decoder (nn.Module): takes encoded and fused AP and LAT view and generates a 3D volume
    """

    def __init__(
        self,
        input_image_size,
        encoder_in_channels,
        encoder_out_channels,
        encoder_strides,
        encoder_kernel_size,
        decoder_in_channels,
        decoder_out_channels,
        decoder_strides,
        decoder_kernel_size,
        ap_expansion_in_channels,
        ap_expansion_out_channels,
        ap_expansion_strides,
        ap_expansion_kernel_size,
        lat_expansion_in_channels,
        lat_expansion_out_channels,
        lat_expansion_strides,
        lat_expansion_kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
        bias,
    ) -> None:
        super().__init__()
        # verify config
        assert (
            len(input_image_size) == 2
        ), f"expected images to be 2D but got {len(input_image_size)}-D"

        self.ap_encoder: nn.Module
        self.lat_encoder: nn.Module
        self.decoder: nn.Module

        self.ap_encoder = nn.Sequential(
            *self._encoder_layer(
                encoder_in_channels,
                encoder_out_channels,
                encoder_strides,
                encoder_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )
        self.lat_encoder = nn.Sequential(
            *self._encoder_layer(
                encoder_in_channels,
                encoder_out_channels,
                encoder_strides,
                encoder_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )

        self.ap_expansion = nn.Sequential(
            *self._expansion_layer(
                ap_expansion_in_channels,
                ap_expansion_out_channels,
                ap_expansion_strides,
                ap_expansion_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )
        self.lat_expansion = nn.Sequential(
            *self._expansion_layer(
                lat_expansion_in_channels,
                lat_expansion_out_channels,
                lat_expansion_strides,
                lat_expansion_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
            )
        )

        self.decoder = nn.Sequential(
            *self._decoder_layers(
                decoder_in_channels,
                decoder_out_channels,
                decoder_strides,
                decoder_kernel_size,
                activation,
                norm,
                dropout,
                dropout_rate,
                bias,
            )
        )

    def _encoder_layer(
        self,
        encoder_in_channels,
        encoder_out_channels,
        encoder_strides,
        encoder_kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
    ):
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            encoder_in_channels,
            encoder_out_channels,
            encoder_strides,
        ):
            layers.append(
                Convolution(
                    spatial_dims=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=encoder_kernel_size,
                    act=activation,
                    norm=norm,
                    dropout=dropout_rate if dropout else None,
                )
            )

        return layers

    def _expansion_layer(
        self,
        in_channels,
        out_channels,
        strides,
        kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
    ):
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            in_channels,
            out_channels,
            strides,
        ):
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=kernel_size,
                    act=activation,
                    norm=norm,
                    dropout=dropout_rate if dropout else None,
                )
            )
        return layers

    def _decoder_layers(
        self,
        decoder_in_channels,
        decoder_out_channels,
        decoder_strides,
        decoder_kernel_size,
        activation,
        norm,
        dropout,
        dropout_rate,
        bias,
    ):
        layers: List[nn.Module] = []
        for index, (in_channels, out_channels, strides, kernel_size) in enumerate(
            zip(
                decoder_in_channels,
                decoder_out_channels,
                decoder_strides,
                decoder_kernel_size,
            )
        ):
            if index == len(decoder_strides) - 1:
                conv_only = True
                # According to `Performance Tuning Guide <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>`_,
                # if a conv layer is directly followed by a batch norm layer, bias should be False.
            else:
                conv_only = False
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=kernel_size,
                    act=activation,
                    norm=norm,
                    dropout=dropout_rate if dropout else None,
                    bias=bias,
                    conv_only=conv_only,
                )
            )
        return layers

    def forward(self, ap_image: torch.Tensor, lat_image: torch.Tensor):
        out_ap = self.ap_encoder(ap_image)
        out_lat = self.lat_encoder(lat_image)

        out_ap_expansion = self.ap_expansion(out_ap.unsqueeze(2))
        out_lat_expansion = self.lat_expansion(out_lat.unsqueeze(-1))

        fused_cube = torch.cat(
            (out_ap_expansion, out_lat_expansion), dim=1
        )  # add new dimension assuming PIR orientation
        return self.decoder(fused_cube)


TwoDPermuteConcatModel = TwoDPermuteConcat

if __name__ == "__main__":
    import math

    import torch

    image_size = 96
    ap_img = torch.zeros((1, 8, image_size, image_size))  # BCHWD
    lat_img = torch.zeros((1, 8, image_size, image_size))
    model_depth = int(math.log2(image_size)) - 2
    config = {
        "input_image_size": [image_size, image_size],
        "encoder": {
            "in_channels": [8, 16, 32, 32, 32, 32],
            "out_channels": [16, 32, 32, 32, 32, 32],
            "strides": [2, 2, 1, 1, 1, 1],
            "kernel_size": 7,
        },
        "ap_expansion": {
            "in_channels": [32, 32, 32, 32, 32, 32, 32][:model_depth],
            "out_channels": [32, 32, 32, 32, 32, 32, 32][:model_depth],
            "strides": ((2, 1, 1),) * model_depth,
            "kernel_size": 3,
        },
        "lat_expansion": {
            "in_channels": [32, 32, 32, 32, 32, 32, 32][:model_depth],
            "out_channels": [32, 32, 32, 32, 32, 32, 32][:model_depth],
            "strides": ((1, 1, 2),) * model_depth,
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
        "bias": False,
    }
    model = TwoDPermuteConcat(config)
    # print(model)
    # fused_cube = model(ap_img, lat_img)
    # print(fused_cube.shape)
    ap_enc_out = model.ap_encoder(ap_img)
    lat_enc_out = model.lat_encoder(lat_img)
    print(ap_enc_out.shape, lat_enc_out.shape)

    ap_after_expansion = model.ap_expansion(ap_enc_out.unsqueeze(2))
    lat_after_expansion = model.lat_expansion(lat_enc_out.unsqueeze(-1))

    print(ap_after_expansion.shape, lat_after_expansion.shape)
    fused_cube = torch.cat(
        (ap_after_expansion, lat_after_expansion), dim=1
    )  # add new dimension assuming PIR orientation
    out = model.decoder(fused_cube)
    print(out.shape)
