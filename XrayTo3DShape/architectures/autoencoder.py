from monai.networks.nets.autoencoder import AutoEncoder
from typing import Sequence, Union, Optional, Tuple, Any
import numpy as np
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
import torch
from torch import nn
import torch.nn.functional as F
import math
import operator
from functools import reduce

# adapted from monai.networks.nets.varautoencoder
# The original Autoencoder in the monai.networks.nets.autoencoder does not have a
# 1D bottlneck latent vector layer i.e. it is fully convolutional.
# In our application we need a low-dimensional  1D embedding of the Biplanar x-rays, hence the need for full connected layer in between the encoder and the decoder


class Encoder1DEmbed(nn.Module):
    """Simple encoder consisting of multiple layers of Convolutions and a fully connected layer in the end to obtain a 1D embedding vector

    Args:
        spatial_dims: number of spatial dimensions
        in_shape: shape of input data with channel dimension
        out_channels: number of output channels
        latent_size: size of the latent variable
        channels: sequence of channels. Top block first. The length of `channels` should be no less than 2.
        strides: sequence of convolution strides. The length of `stride` should equal to `len(channels) - 1`.
        kernel_size: convolution kernel size, the value(s) shoulde be odd.
        num_res_units: number of residual units. Defaults to 0.
        act: activation type and arguments.
        norm: feature normalization type and arguments.
        dropout: dropout ration.
        bias: whether to have a bias term in convolution blocks. Defaults to true. According to Performance Tuning Guide, if a conv layer is directly followed by a batch norm layer, bias should be False.

    Examples::

        net = Encoder(
            spatial_dims=2,
            in_shape=(1,64,64),
            out_channels=1,
            latent_size = 64,
            channels=(2,4,8),
            strides=(2,2,2)
        )
    """

    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Optional[str] = "RELU",
        norm: Union[Tuple, str] = "INSTANCE",
        dropout: Optional[float] = None,
        bias: bool = True,
    ) -> None:
        in_channels, *self.in_shape = in_shape
        self.latent_size = latent_size

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias

        if len(channels) != len(strides):
            raise ValueError("Encoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        self.encode, self.encoded_channels = self._get_encode_module(
            self.encoded_channels, channels, strides
        )

        padding = same_padding(self.kernel_size)
        self.final_size = np.asarray(self.in_shape,dtype=int)
        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, self.kernel_size, s, padding)

        linear_size = int(np.product(self.final_size)) * self.encoded_channels
        self.latent_encode_layer = nn.Linear(linear_size,self.latent_size)


    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> Tuple[nn.Sequential, int]:
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module(f"encode_{i}", layer)
            layer_channels = c
        return encode, layer_channels

    def _get_encode_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> nn.Module:
        """
        Returns a single layer of the encoder part of the network.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_last,
            )
            return mod
        mod = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_last,
        )
        return mod

    def forward(self, x: torch.Tensor) -> Any:
        x = self.encode(x)
        x = x.view(x.shape[0],-1)
        x = self.latent_encode_layer(x)
        return x


class AutoEncoder1DEmbed(AutoEncoder):
    def __init__(
        self,
        spatial_dims: int,
        in_shape: Sequence[int],
        out_channels: int,
        latent_size: int,
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
        use_sigmoid: bool = True,
    ) -> None:
        self.in_channels, *self.in_shape = in_shape
        self.use_sigmoid = use_sigmoid

        self.latent_size = latent_size
        self.final_size = np.asarray(self.in_shape, dtype=int)

        super().__init__(
            spatial_dims,
            self.in_channels,
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

        padding = same_padding(self.kernel_size)

        for s in strides:
            self.final_size = calculate_out_shape(self.final_size, self.kernel_size, s, padding)

        linear_size = int(np.product(self.final_size)) * self.encoded_channels
        self.latent_encode_layer = nn.Sequential(nn.Linear(linear_size, self.latent_size),
                                                 nn.InstanceNorm1d(num_features=self.latent_size),
                                                 nn.Tanh())
        self.latent_decoder_layer = nn.Sequential(nn.Linear(self.latent_size, linear_size),
                                                  nn.InstanceNorm1d(num_features=linear_size),
                                                  nn.ReLU())
        self._initialize_weights()

    def encode_forward(self, x: torch.Tensor):
        x = self.encode(x)
        x = self.intermediate(x)
        x = x.view(x.shape[0], -1)
        x = self.latent_encode_layer(x)
        return x

    def decode_forward(self, x: torch.Tensor, use_sigmoid: bool = True) -> torch.Tensor:
        x = self.latent_decoder_layer(x)
        x = x.view(x.shape[0], self.channels[-1], *self.final_size)
        x = self.decode(x)
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> Any:
        latent_vector = self.encode_forward(x)
        return self.decode_forward(latent_vector),latent_vector

    def _initialize_weights(self)-> None:
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

if __name__ == "__main__":
    model = AutoEncoder1DEmbed(
        spatial_dims=2,
        in_shape=(1, 64, 64),
        out_channels=1,
        latent_size=64,
        channels=(16, 32, 64),
        strides=(1, 2, 2),
    )
    out = model(torch.zeros((1, 1, 64, 64)))
    print(model)
    print(out.shape)

    encoder_model = Encoder1DEmbed(
        spatial_dims=2,
        in_shape=(1, 64, 64),
        out_channels=1,
        latent_size=64,
        channels=(2, 4, 8),
        strides=(2, 2, 2),
    )
    print(encoder_model)
    out = encoder_model(torch.zeros((1, 1, 64, 64)))
    print(out.shape)
