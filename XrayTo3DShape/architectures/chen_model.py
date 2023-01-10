import torch
from torch import nn
from typing import List, Dict
from monai.networks.blocks.convolutions import Convolution


class OneDConcatModel(nn.Module):
    """
    The idea of this model is to encode each of the PA and LAT images into 1D vectors.
    These are then concatenated and decoded into a 3D segmentation.
    Chen, Chih-Chia, and Yu-Hua Fang. "Using bi-planar x-ray images to reconstruct the spine structure by the convolution neural network." International Conference on Biomedical and Health Informatics. Springer, Cham, 2019.
    Attributes:
        pa_encoder (nn.Module): Encodes the AP image
        lat_encoder (nn.Module): Encodes the LAT image
        reshape_module (nn.Module): reshape the 1D vector into a (1,1,1) cube with number of channels equal to length of the 1D vector
        decoder (nn.Module): decodes the fused cube into a full-fledged volume
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.ap_encoder: nn.Module
        self.lat_encoder: nn.Module
        self.reshape_module: nn.Module
        self.decoder: nn.Module

        assert self._calculate_1d_vec_channels() == self.config['decoder']['in_channels'][0], f"expected {self._calculate_1d_vec_channels()}, got {self.config['decoder']['in_channels'][0]}"
        self.ap_encoder = nn.Sequential(*self._encoder_layer())
        self.lat_encoder = nn.Sequential(*self._encoder_layer())
        self.decoder = nn.Sequential(*self._decoder_layers())

    def _encoder_layer(self):
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            self.config["encoder"]["in_channels"],
            self.config["encoder"]["out_channels"],
            self.config["encoder"]["strides"],
        ):
            layers.append(
                Convolution(
                    spatial_dims=2,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=self.config["kernel_size"],
                    act=self.config["act"],
                    norm=self.config["norm"],
                    dropout=self.config["dropout"],
                )
            )

        layers.append(nn.Flatten())
        return layers

    def _decoder_layers(self):
        layers: List[nn.Module] = []
        for in_channels, out_channels, strides in zip(
            self.config["decoder"]["in_channels"],
            self.config["decoder"]["out_channels"],
            self.config["decoder"]["strides"],
        ):
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=self.config["kernel_size"],
                    act=self.config["act"],
                    norm=self.config["norm"],
                    dropout=self.config["dropout"],
                )
            )
        return layers

    def _calculate_1d_vec_channels(self)->int:
        """calculate size of the 1D Low-dim embedding for given model layers"""
        input_width,input_height = self.config['input_image_size']
        output_width, output_height = input_width, input_height
        for stride in self.config['encoder']['strides']:
            output_width /= stride
            output_height /= stride
        ap_encoder_1d_vecsize = output_height * output_height * self.config['encoder']['out_channels'][-1]
        lat_encoder_1d_vecsize = ap_encoder_1d_vecsize
        decoded_cube_channels = ap_encoder_1d_vecsize  + lat_encoder_1d_vecsize

        return int(decoded_cube_channels)

    def forward(self, ap_image: torch.Tensor, lat_image: torch.Tensor):
        out_ap = self.ap_encoder(ap_image)
        out_lat = self.lat_encoder(lat_image)

        # concatenate along channel dimension BCHWD
        fused_1d_vec = torch.cat([out_ap, out_lat], dim=1)
        fused_cube = fused_1d_vec.view(size=(-1,self._calculate_1d_vec_channels(),*(1,1,1)))

        # decoder
        out_decoder = self.decoder(fused_cube)
        return out_decoder


