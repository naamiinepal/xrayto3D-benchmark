# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""Atlas-deformation based encoder-decoder architecture
TODO: work in progress
"""
from typing import Dict, List

import torch
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act
from torch import nn


class AtlasDeformationSTN(nn.Module):
    """atlas-deformation based encoder-decoder"""
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

        self.ap_encoder = nn.Sequential(*self._encoder_layer())
        self.lat_encoder = nn.Sequential(*self._encoder_layer())

        self.ap_expansion = nn.Sequential(*self._expansion_layer("ap"))
        self.lat_expansion = nn.Sequential(*self._expansion_layer("lat"))
        self.affine_decoder = nn.Sequential(*self._affine_decoder_layer())

    def _affine_decoder_layer(self):
        layers: List[nn.Module] = []

        layers.append(Convolution(spatial_dims=3, in_channels=65, out_channels=4))
        layers.append(nn.Flatten())

        for in_channels, out_channels in zip(
            self.config["affine"]["in_channels"], self.config["affine"]["out_channels"]
        ):
            layers.append(nn.Linear(in_channels, out_channels))

            act_type = Act[self.config["act"]]
            layers.append(act_type())

        layers.append(
            nn.Linear(self.config["affine"]["out_channels"][-1], 3 * 4)
        )  # generate affine matrix
        return layers

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

    def _expansion_layer(self, image_type_suffix: str):
        assert image_type_suffix in [
            "ap",
            "lat",
        ], "image type should be one of AP or LAT"

        expansion_type = f"{image_type_suffix}_expansion"
        layers: List[nn.Module] = []

        for in_channels, out_channels, strides in zip(
            self.config[expansion_type]["in_channels"],
            self.config[expansion_type]["out_channels"],
            self.config[expansion_type]["strides"],
        ):
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

    def forward(self, ap, lat, atlas_seg):
        atlas_seg_scaled = F.interpolate(
            atlas_seg.clone(), scale_factor=1.0 / 4.0, mode="nearest"
        )

        out_ap = self.ap_encoder(ap)
        out_lat = self.lat_encoder(lat)
        fused_cube = torch.cat(
            (
                self.ap_expansion(out_ap.unsqueeze(2)),
                self.lat_expansion(out_lat.unsqueeze(-1)),
            ),
            dim=1,
        )  # add new dimension assuming PIR orientation
        # print('fused cube',fused_cube.shape,'atlas seg',atlas_seg.shape)
        out = torch.cat(
            [fused_cube, atlas_seg_scaled], dim=1
        )  # concatenate along channels
        out = self.affine_decoder(out)
        # encoder_out = torch.cat([self.ap_encoder(ap), self.lat_encoder(lat)],dim=1)
        # affine_in = self.affine_decoder(encoder_out)
        theta = out.view(-1, 3, 4)
        affine_grid = F.affine_grid(theta, atlas_seg.size(), align_corners=True)
        return F.grid_sample(atlas_seg, affine_grid, align_corners=True)
        # return out


if __name__ == "__main__":
    config = {
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
        "affine": {
            "in_channels": [16384, 4096, 1024],
            "out_channels": [
                4096,
                1024,
                32,
            ],
        },
        "kernel_size": 5,
        "act": "RELU",
        "norm": "BATCH",
        "dropout": 0.0,
    }
    model = AtlasDeformationSTN(config)
    in_tensor = torch.zeros((1, 1, 64, 64))
    atlas_seg_tensor = torch.zeros(1, 1, 64, 64, 64)

    atlas_seg_scaled = F.interpolate(
        atlas_seg_tensor.clone(), scale_factor=1.0 / 4.0, mode="nearest"
    )

    from XrayTo3DShape import printarr

    out = model(in_tensor, in_tensor, atlas_seg_tensor)
    printarr(out, in_tensor, atlas_seg_tensor, atlas_seg_scaled)
