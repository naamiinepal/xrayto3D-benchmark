from monai.networks.nets.autoencoder import AutoEncoder
from typing import Sequence, Union, Optional, Tuple, Any
import numpy as np
from monai.networks.layers.convutils import calculate_out_shape, same_padding
import torch
from torch import nn
import torch.nn.functional as F

# adapted from monai.networks.nets.varautoencoder
# The original Autoencoder in the monai.networks.nets.autoencoder does not have a
# 1D bottlneck latent vector layer i.e. it is fully convolutional.
# In our application we need a low-dimensional  1D embedding of the Biplanar x-rays, hence the need for full connected layer in between the encoder and the decoder

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
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Union[Tuple, str] = "BATCH",
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
        self.latent_encode_layer = nn.Linear(linear_size,self.latent_size)
        self.latent_decoder_layer = nn.Linear(self.latent_size, linear_size)

    def encode_forward(self, x:torch.Tensor):
        x = self.encode(x)
        x = self.intermediate(x)
        x = x.view(x.shape[0],-1)
        x = self.latent_encode_layer(x)
        return x
    
    def decode_forward(self, x:torch.Tensor, use_sigmoid:bool = True)-> torch.Tensor:
        x = F.relu(self.latent_decoder_layer(x))
        x = x.view(x.shape[0], self.channels[-1],*self.final_size)
        x = self.decode(x)
        if use_sigmoid:
            x = torch.sigmoid(x)
        return x

    def forward(self, x: torch.Tensor) -> Any:
        return self.decode_forward(self.encode_forward(x))

if __name__ == "__main__":
    model = AutoEncoder1DEmbed(
        spatial_dims=2,
        in_shape=(1, 64, 64),
        out_channels=1,
        latent_size=64,
        channels=(16, 32, 64),
        strides=(1, 2, 2),
    )
    out = model(torch.zeros((1,1,64,64)))
    print(model)
    print(out.shape)
