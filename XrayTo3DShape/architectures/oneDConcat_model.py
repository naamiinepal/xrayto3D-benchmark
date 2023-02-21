import torch
from torch import nn
from typing import List, Dict
from monai.networks.blocks.convolutions import Convolution
from .utils import calculate_1d_vec_channels

class OneDConcat(nn.Module):
    """
    The idea of this model is to encode each of the PA and LAT images into 1D vectors.
    These are then concatenated and decoded into a 3D segmentation.
    Chen, Chih-Chia, and Yu-Hua Fang. "Using bi-planar x-ray images to reconstruct the spine structure by the convolution neural network." International Conference on Biomedical and Health Informatics. Springer, Cham, 2019.
    Attributes:
        pa_encoder (nn.Module): Encodes the AP image
        lat_encoder (nn.Module): Encodes the LAT image
        decoder (nn.Module): decodes the fused cube into a full-fledged volume
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.ap_encoder: nn.Module
        self.lat_encoder: nn.Module
        self.decoder: nn.Module
        self.bottlneck_size = config['bottleneck_size']
        # verify config
        assert len(self.config['input_image_size']) == 2, f'expected images to be 2D but got {len(self.config["input_image_size"])}D'
        # assert self._calculate_1d_vec_channels() == self.config['decoder']['in_channels'][0], f"expected {self._calculate_1d_vec_channels()}, got {self.config['decoder']['in_channels'][0]}"


        self.ap_encoder = nn.Sequential(*self._encoder_layer())
        self.lat_encoder = nn.Sequential(*self._encoder_layer())
        self.fully_connected_bottlneck = nn.Sequential(nn.Linear(in_features=self._calculate_1d_vec_channels(),out_features=self.bottlneck_size),nn.BatchNorm1d(num_features=config['bottleneck_size']),nn.ReLU())
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
        for index, (in_channels, out_channels, strides) in enumerate(zip(
            self.config["decoder"]["in_channels"],
            self.config["decoder"]["out_channels"],
            self.config["decoder"]["strides"],
        )):
            # the last layer does not have activation
            if index == len(self.config['decoder']['strides']) - 1:
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
                    kernel_size=self.config["kernel_size"],
                    act=self.config['act'],
                    norm=self.config["norm"],
                    dropout=self.config["dropout"],
                    bias=self.config['bias'],
                    conv_only=conv_only
                )
            )
        return layers

    def _calculate_1d_vec_channels(self)->int:
        image_size = self.config['input_image_size']
        encoder_strides = self.config['encoder']['strides']
        encoder_out_channel = self.config['encoder']['out_channels'][-1]
        return calculate_1d_vec_channels(image_size[0],encoder_strides,encoder_out_channel)
        
    def forward(self, ap_image: torch.Tensor, lat_image: torch.Tensor):
        out_ap = self.ap_encoder(ap_image)
        out_lat = self.lat_encoder(lat_image)

        # concatenate along channel dimension BCHWD
        fused_1d_vec = torch.cat([out_ap, out_lat], dim=1)
        # print(f'fused 1d vec {fused_1d_vec.shape}')
        embedding_vector = self.fully_connected_bottlneck(fused_1d_vec)
        # print(f'embedding vector {embedding_vector.shape}')
        fused_cube = embedding_vector.view(size=(-1,self.bottlneck_size,*(1,1,1)))

        # decoder
        out_decoder = self.decoder(fused_cube)
        return out_decoder


