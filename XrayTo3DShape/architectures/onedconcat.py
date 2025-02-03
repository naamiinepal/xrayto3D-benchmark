# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


from typing import Dict, List

import torch
from monai.networks.blocks.convolutions import Convolution
from torch import nn
from .arch_utils import calculate_1d_vec_channels
from ..utils.registry import ARCHITECTURES


@ARCHITECTURES.register("OneDConcatModel")
class OneDConcat(nn.Module):
    """
    The idea of this model is to encode each of the PA and LAT images into 1D vectors.
    These are then concatenated and decoded into a 3D segmentation.
    Chen, Chih-Chia, and Yu-Hua Fang. "Using bi-planar x-ray images to reconstruct vertebra
    by the convolution neural network.
    " International Conference on Biomedical and Health Informatics. Springer, Cham, 2019.
    Attributes:
        pa_encoder (nn.Module): Encodes the AP image
        lat_encoder (nn.Module): Encodes the LAT image
        decoder (nn.Module): decodes the fused cube into a full-fledged volume
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
        activation,
        norm,
        dropout,
        dropout_rate,
        bottleneck_size,
        bias
    ) -> None:
        super().__init__()
        self.ap_encoder: nn.Module
        self.lat_encoder: nn.Module
        self.decoder: nn.Module
        self.bottlneck_size = bottleneck_size
        # verify config
        assert (
            len(input_image_size) == 2
        ), f"expected images to be 2D but got {len(input_image_size)}-D"

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
                bias
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
                bias
            )
        )
        self.fully_connected_bottlneck = nn.Sequential(
            nn.Linear(
                in_features=self._calculate_1d_vec_channels(
                    input_image_size,encoder_strides,encoder_out_channels
                ),
                out_features=bottleneck_size,
            ),
            nn.BatchNorm1d(num_features=bottleneck_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(*self._decoder_layers(
            decoder_in_channels,decoder_out_channels,decoder_strides,decoder_kernel_size,activation,norm,dropout,dropout_rate,bias
        ))

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
        bias
    ):
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            encoder_in_channels, encoder_out_channels, encoder_strides
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
                    bias=bias
                )
            )

        layers.append(nn.Flatten())
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
        bias
    ):
        layers: List[nn.Module] = []
        for index, (in_channels, out_channels, strides) in enumerate(
            zip(decoder_in_channels, decoder_out_channels, decoder_strides)
        ):
            # the last layer does not have activation
            if index == len(decoder_strides) - 1:
                conv_only = True
            else:
                conv_only = False
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=decoder_kernel_size,
                    act=activation,
                    norm=norm,
                    dropout=dropout_rate if dropout else None,
                    bias=bias,
                    conv_only=conv_only,
                )
            )
        return layers

    def _calculate_1d_vec_channels(self,input_image_size,encoder_strides,encoder_out_channels) -> int:
        image_width, image_height = input_image_size
        assert image_width == image_height, f'expected image size {input_image_size} to be square'
        
        encoder_strides = encoder_strides
        encoder_last_channel = encoder_out_channels[-1]
        return calculate_1d_vec_channels(
            spatial_dims=2,
            image_size=image_width,
            encoder_strides=encoder_strides,
            encoder_last_channel=encoder_last_channel,
        )

    def forward(self, ap_image: torch.Tensor, lat_image: torch.Tensor):
        out_ap = self.ap_encoder(ap_image)
        out_lat = self.lat_encoder(lat_image)

        # concatenate along channel dimension BCHWD
        fused_1d_vec = torch.cat([out_ap, out_lat], dim=1)
        # print(f'fused 1d vec {fused_1d_vec.shape}')
        embedding_vector = self.fully_connected_bottlneck(fused_1d_vec)
        # print(f'embedding vector {embedding_vector.shape}')
        fused_cube = embedding_vector.view(size=(-1, self.bottlneck_size, *(1, 1, 1)))

        # decoder
        out_decoder = self.decoder(fused_cube)
        return out_decoder


OneDConcatModel = OneDConcat
