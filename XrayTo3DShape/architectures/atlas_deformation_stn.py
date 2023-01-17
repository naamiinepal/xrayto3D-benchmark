import torch
from torch import nn
from typing import Dict, Sequence, Tuple,List
from monai.networks.blocks.convolutions import Convolution,ResidualUnit
from monai.networks.blocks.upsample import Upsample
from monai.networks.blocks.mlp import MLPBlock
from monai.networks.layers import convutils
import einops
import torch.nn.functional as F
from monai.networks.layers.factories import Act

class AtlasDeformationSTN(nn.Module):
    def __init__(self,config: Dict) -> None:
        super().__init__()
        self.config = config      

        self.ap_encoder = nn.Sequential(*self._encoder_layer())
        self.lat_encoder = nn.Sequential(*self._encoder_layer())

        self.affine_decoder = nn.Sequential(*self._affine_decoder_layer())
    
    def _affine_decoder_layer(self):
        layers: List[nn.Module] = []

        layers.append(nn.Flatten())

        for in_channels, out_channels in zip(
            self.config['affine']['in_channels'],
            self.config['affine']['out_channels']
        ):
            layers.append(nn.Linear(in_channels,out_channels))
            
            act_type = Act[self.config['act']]
            layers.append(act_type())

        layers.append(nn.Linear(self.config['affine']['out_channels'][-1],3*4)) # generate 2x3 affine matrix
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
                    kernel_size=self.config["kernel_size"],
                    act=self.config["act"],
                    norm=self.config["norm"],
                    dropout=self.config["dropout"],
                )
            )

        return layers

    def forward(self,ap,lat,atlas_seg):
        encoder_out = torch.cat([self.ap_encoder(ap), self.lat_encoder(lat)],dim=1)
        affine_in = self.affine_decoder(encoder_out)
        theta = affine_in.view(-1,3,4)
        affine_grid = F.affine_grid(theta,atlas_seg.size())
        return F.grid_sample(atlas_seg,affine_grid)


if __name__ == '__main__':
    config = {'encoder':{
        'in_channels' : [1,8,16],
        'out_channels' : [8,16,32],
        'strides': [2,2,2] 
        },
        'affine': {
            'in_channels':[4096,1024],
            'out_channels': [1024,32,]
        },
        'kernel_size':5,
        'act': 'RELU',
        'norm': 'BATCH',
        'dropout' : 0.0,
    }
    model = AtlasDeformationSTN(config)
    in_tensor = torch.zeros((1,1,64,64))
    atlas_seg_tensor = torch.zeros(1,1,64,64,64)
    out = model(in_tensor,in_tensor,atlas_seg_tensor)
    print(out.shape)
