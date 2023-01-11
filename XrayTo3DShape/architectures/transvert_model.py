import torch
from torch import nn
from typing import Dict, List
from monai.networks.blocks.convolutions import Convolution


class TwoDPermuteConcat(nn.Module):
    """
    Transvert Architecture
    Bayat, Amirhossein, et al. "Inferring the 3D standing spine posture from 2D radiographs.", 2020.

        Attributes:
        pa_encoder (nn.Module): encodes AP view x-rays
        lat_encoder (nn.Module): encodes LAT view x-rays
        decoder (nn.Module): takes encoded and fused AP and LAT view and generates a 3D volume
    """

    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config
        self.ap_encoder: nn.Module
        self.lat_encoder: nn.Module
        self.decoder: nn.Module

        self.ap_encoder = nn.Sequential(*self._encoder_layer())
        self.lat_encoder = nn.Sequential(*self._encoder_layer())

        self.ap_expansion = nn.Sequential(*self._expansion_layer('ap'))
        self.lat_expansion = nn.Sequential(*self._expansion_layer('lat'))

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
                    kernel_size=self.config["encoder"]["kernel_size"],
                    act=self.config["act"],
                    norm=self.config["norm"],
                    dropout=self.config["dropout"],
                )
            )

        return layers

    def _expansion_layer(self,image_type_suffix:str):
        assert image_type_suffix in ['ap','lat'], f'image type should be one of AP or LAT'

        expansion_type = f'{image_type_suffix}_expansion'
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            self.config[expansion_type]["in_channels"],self.config[expansion_type]["out_channels"],self.config[expansion_type]["strides"]
        ):
            print(in_channels, out_channels, strides)
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=self.config[expansion_type]["kernel_size"],
                    act=self.config["act"],
                    norm=self.config["norm"],
                    dropout=self.config["dropout"],
                )
            )
        return layers
    
    def _decoder_layers(self):
        layers: List[nn.Module] = []
        for in_channels, out_channels, strides,kernel_size in zip(
            self.config['decoder']["in_channels"],self.config['decoder']["out_channels"],self.config['decoder']["strides"], self.config['decoder']['kernel_size']
        ):
            print(in_channels, out_channels, strides)
            layers.append(
                Convolution(
                    spatial_dims=3,
                    is_transposed=True,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    strides=strides,
                    kernel_size=kernel_size,
                    act=self.config["act"],
                    norm=self.config["norm"],
                    dropout=self.config["dropout"],
                )
            )        
        return layers

    def forward(self, ap_image: torch.Tensor, lat_image: torch.Tensor):
        out_ap = self.ap_encoder(ap_image)
        out_lat = self.lat_encoder(lat_image)

        # out_ap = self.ap_expansion(out_ap.unsqueeze(0))
        fused_cube = torch.concat((self.ap_expansion(out_ap.unsqueeze(2)), self.lat_expansion(out_lat.unsqueeze(-1))),dim=1) # add new dimension assuming PIR orientation
        return self.decoder(fused_cube)


if __name__ == "__main__":
    import torch

    ap_img = torch.zeros((1, 8, 64, 64))  # BCHWD
    lat_img = torch.zeros((1, 8, 64, 64))
    config = {
        "input_image_size": [64, 64],
        "encoder": {
            "in_channels": [8, 16, 32, 32, 32, 32],
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
    }
    model = TwoDPermuteConcat(config)
    print(model)
    fused_cube = model(ap_img, lat_img)
    print(fused_cube.shape)
