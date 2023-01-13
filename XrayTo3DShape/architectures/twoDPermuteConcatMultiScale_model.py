import torch
from torch import nn
from typing import Dict
from .dynunet_mod import *
from monai.networks.blocks.convolutions import Convolution

class TwoDPermuteConcatMultiScale(nn.Module):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        self.config = config

        in_ch, h, w = self.config["in_shape"]
        assert h == w, f"Height and width of input should be equal but got height {h} and weight {w}"

        self.ap_encoder: nn.Module
        self.lat_encoder: nn.Module
        self.decoder: nn.ModuleList
        self.segmentation_head: nn.Module

        self.ap_encoder = DynUnetNoInterp(
            spatial_dims=2,
            in_shape=self.config["in_shape"],
            out_channels=h,
            kernel_size=self.config['encoder']["kernel_size"],
            strides=self.config['encoder']["strides"],
            upsample_kernel_size=self.config['encoder']["strides"][1:],
            deep_supervision=True,
            deep_supr_num=len(self.config['encoder']['strides']) - 2,
            interpolate_flag=False,
        )

        self.lat_encoder = DynUnetNoInterp(
            spatial_dims=2,
            in_shape=self.config["in_shape"],
            out_channels=h,
            kernel_size=self.config['encoder']["kernel_size"],
            strides=self.config['encoder']["strides"],
            upsample_kernel_size=self.config['encoder']["strides"][1:],
            deep_supervision=True,
            deep_supr_num=len(self.config['encoder']['strides']) - 2,
            interpolate_flag=False,
        ) 

        self.decoder =  nn.ModuleList(self.get_decoder_block(out_channel=self.config['decoder']['out_channel']))
        self.segmentation_head = Convolution(
            spatial_dims=3,
            in_channels=self.config['decoder']['out_channel'],
            out_channels=1,
            strides=1,
            act=self.config['act'],
            norm=self.config['norm'],
            is_transposed=False,
            conv_only=True,
        )

    def get_decoder_block(self,out_channel:int):
        decode_layer = []
        num_deep_supr_blocks = len(self.config['encoder']['strides']) - 2
        for index in range(num_deep_supr_blocks):
            # the first decoder layer gets ap and lat cubes
            # the subsequent layers get the volume from previous layer and the ap+lat cubes
            in_ch = 2 if index == 0 else out_channel + 2 
            decode_layer.append(
                Convolution(
                    spatial_dims=3,
                    in_channels=in_ch,
                    out_channels=out_channel,
                    strides=2,
                    kernel_size=self.config['decoder']['kernel_size'],
                    act=self.config['act'],
                    norm=self.config['norm'],
                    is_transposed=True,
                    conv_only=False,
                )
            )
        return decode_layer
    
    def forward(self,ap_image:torch.Tensor, lat_image: torch.Tensor):
        out_ap:list[torch.Tensor] = self.ap_encoder(ap_image)
        out_lat:list[torch.Tensor] = self.lat_encoder(lat_image)

        x = torch.rand(0)
        # x: torch.Tensor

        # multiscale fuse and decode
        for index, (ap_cube, lat_cube, decoder_layer) in enumerate(
            zip(out_ap[::-1], out_lat[::-1], self.decoder)
        ):
            # assume a PIR orientation and standing AP and LAT views.
            # permute the LAT orientation so that the last dim represents Left to Right orientation

            permuted_lat_cube = torch.swapdims(lat_cube, 1, -1)
            fused_cube = torch.stack((ap_cube, permuted_lat_cube), dim=1)  # 2 channels
            if index == 0:
                x = decoder_layer(fused_cube)
            else:
                x = decoder_layer(torch.cat((fused_cube, x), dim=1))        

        return self.segmentation_head(x)
        
if __name__ == "__main__":
    config = {
        "in_shape": (1,64,64),
        "kernel_size":3,
        "act":'RELU',
        "norm":"BATCH",
        "encoder":{
            "kernel_size":(3,)*4,
            "strides":(1,2,2,2),   # keep the first element of the strides 1 so the input and output shape match

        },
        "decoder": {
            "out_channel":16,
            "kernel_size":3
        }
    }



    x_ray_img = torch.zeros(1, *config['in_shape'])

    model = TwoDPermuteConcatMultiScale(config)
    out = model(x_ray_img,x_ray_img)
    print(out.shape)








    # print([ o.shape for o in ap_out])
    from torchview import draw_graph

    draw_graph(
        ap_unet,
        input_data=torch.zeros(1, 1, 64, 64),
        graph_name="DynUnet",
        save_graph=True,
        filename="docs/arch_viz/dynunet",
        depth=5,
    )
