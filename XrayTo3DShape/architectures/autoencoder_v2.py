# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""
adapted from monai.networks.nets.varautoencoder
The original Autoencoder in the monai.networks.nets.autoencoder does not have a
1D bottlneck latent vector layer i.e. it is fully convolutional.
In our application we need a low-dimensional  1D embedding of the Biplanar x-rays,
hence the need for full connected layer in between the encoder and the decoder
TODO: investigate loss in reconstruction when using 1D-latent vector instead of fully conv-AE
"""
import math
import operator
from functools import reduce
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.nets.autoencoder import AutoEncoder
from torch import nn
from ..utils.registry import ARCHITECTURES

@ARCHITECTURES.register('CustomAutoEncoder')
class CustomAutoEncoder(AutoEncoder):
    """Simple encoder consisting of multiple layers of Convolutions
    and a fully connected layer in the end to obtain a 1D embedding vector
    """

    def __init__(
        self,
        latent_dim: int,
        image_size: int,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        inter_channels: Optional[list] = None,
        inter_dilations: Optional[list] = None,
        num_inter_units: int = 2,
        act: Optional[Union[Tuple, str]] = "RELU",
        norm: Union[Tuple, str] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
    ) -> None:
        super().__init__(
            spatial_dims,
            in_channels,
            out_channels,
            channels,
            strides,
            kernel_size,
            up_kernel_size,
            num_res_units,
            inter_channels,
            inter_dilations,
            num_inter_units,
            act,
            norm,
            dropout,
            bias,
        )

        self.enc_conv_out_shape = image_size
        for stride in strides:
            self.enc_conv_out_shape = calculate_out_shape(
                in_shape=self.enc_conv_out_shape,
                kernel_size=kernel_size,
                stride=stride,
                padding=same_padding(kernel_size, dilation=1),
            )
        self.enc_conv_out_shape = self.enc_conv_out_shape * spatial_dims

        self.bottleneck_fcn_encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod((channels[-1], *self.enc_conv_out_shape)), latent_dim),
        )

        self.bottleneck_fcn_decode = nn.Sequential(
            nn.Linear(latent_dim, np.prod((channels[-1], *self.enc_conv_out_shape)))
        )

        self._initialize_weights()

    def latent_vec_decode(self, latent_vec):
        x = self.bottleneck_fcn_decode(latent_vec)
        x = x.reshape(x.shape[0], -1, *self.enc_conv_out_shape)
        pred_logits = self.decode(x)
        return pred_logits

    def forward(self, x):
        x = self.encode(x)
        latent_vec = self.bottleneck_fcn_encode(x)
        x = self.latent_vec_decode(latent_vec)
        return x, latent_vec

    def _initialize_weights(self) -> None:
        """
        Args:
            None, initializes weights for conv/linear/batchnorm layers
            following weight init methods from
            `official Tensorflow EfficientNet implementation
            <https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/efficientnet_model.py#L61>`_.
            Adapted from `EfficientNet-PyTorch's init method
            <https://github.com/rwightman/gen-efficientnet-pytorch/blob/master/geffnet/efficientnet_builder.py>`_.
        """
        for name, m in self.named_modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_out = reduce(operator.mul, m.kernel_size, 1) * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                fan_out = m.weight.size(0)
                fan_in = 0
                init_range = 1.0 / math.sqrt(fan_in + fan_out)
                m.weight.data.uniform_(-init_range, init_range)
                m.bias.data.zero_()

@ARCHITECTURES.register('TLPredictor')
class TLPredictor(nn.Module):
    """Generate 2D Embedding vector"""

    def __init__(
        self,
        image_size: int,
        spatial_dims: int,
        in_channel: int,
        channels: Sequence[int],
        strides: Sequence[int],
        latent_dim: int,
        kernel_size: Union[Sequence[int], int] = 3,
        act: Optional[Union[Tuple, str]] = "RELU",
        norm: Union[Tuple, str] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        bias: bool = True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.enc_conv_out_shape = image_size
        for stride in strides:
            self.enc_conv_out_shape = calculate_out_shape(
                in_shape=self.enc_conv_out_shape,
                kernel_size=kernel_size,
                stride=stride,
                padding=same_padding(kernel_size, dilation=1),
            )
        self.enc_conv_out_shape = self.enc_conv_out_shape * spatial_dims

        encode_layers = []
        layer_channel = in_channel
        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = Convolution(
                spatial_dims=spatial_dims,
                in_channels=layer_channel,
                out_channels=c,
                strides=s,
                kernel_size=kernel_size,
                act=act,
                norm=norm,
                bias=bias,
                dropout=dropout,
            )
            layer_channel = c
            encode_layers.append(layer)

        self.model = nn.Sequential(
            *encode_layers,
            nn.Flatten(),
            nn.Linear(np.prod((channels[-1], *self.enc_conv_out_shape)), latent_dim)
        )

    def forward(self, x):
        return self.model(x)
