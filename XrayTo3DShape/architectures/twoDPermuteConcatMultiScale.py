from torch import nn
import torch
from monai.networks.blocks.convolutions import Convolution
from typing import Dict, List

class DenseNetBlock(nn.Module):
    def __init__(self,in_channel = 8,out_channel = 16,encoder_count=3) -> None:
        super().__init__()
        encoders_list =  [Convolution(
            spatial_dims=2,
            in_channels=in_channel+i*out_channel,
            out_channels=out_channel,
            act='RELU',norm='BATCH') for i in range(encoder_count)]
        self.encoders = nn.Sequential(*encoders_list)
    
    def forward(self,x:torch.Tensor):
        x_concat = x
        for i, encoder in enumerate(self.encoders):
            x = encoder(x_concat)
            x_concat = torch.cat((x_concat,x),dim=1)
        return x_concat

class MultiScaleEncoder2d(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 =  Convolution(spatial_dims=2,in_channels=1,out_channels=16,strides=1,padding=1,act='RELU',norm='BATCH')
        self.multiscale_encoder = nn.ModuleList([
            DenseNetBlock(in_channel=16,out_channel=4,encoder_count=4), # (B,16,H,W)->(B,32,H,W) 32 = 16 + 4 + 4 + 4 + 4,
            DenseNetBlock(in_channel=32,out_channel=8,encoder_count=4) # (B,32,H/2,W/2)->(B,64,H,W) 64 = 32 + 8 + 8 + 8 + 8
            ]
        )
        self.downsampler = nn.MaxPool2d(kernel_size=2,stride=2,padding=0,return_indices=False)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        multiscale_x = []
        for i, layer in enumerate(self.multiscale_encoder):
            x = layer(x)
            multiscale_x.insert(0,x) # prepend
            if i != len(self.multiscale_encoder) - 1:
                x = self.downsampler(x)
        return multiscale_x

class MultiScale2DPermuteConcat(nn.Module):
    def __init__(self,permute:bool=True) -> None:
        super().__init__()
        self.permute = permute
        self.ap_encoder :nn.Module
        self.lat_encoder : nn.Module

        self.ap_encoder = MultiScaleEncoder2d()
        self.lat_encoder = MultiScaleEncoder2d()

        self.ap_decoder_pipeline : nn.ModuleList
        self.lat_decoder_pipeline : nn.ModuleList
        self.fusion_decoder_pipeline : nn.ModuleList

        ap_decoder_list = [
            Convolution(spatial_dims=2,in_channels=64,out_channels=64,kernel_size=3,strides=1,padding=1,act='RELU',norm='BATCH'),

            Convolution(spatial_dims=2,in_channels=32+64,out_channels=128,kernel_size=3,strides=1,padding=1,act='RELU',norm='BATCH'),


        ]
        self.ap_decoder_pipeline = nn.ModuleList(ap_decoder_list)

        lat_decoder_list = [
            Convolution(spatial_dims=2,in_channels=64,out_channels=64,kernel_size=3,strides=1,padding=1,act='RELU',norm='BATCH'),

            Convolution(spatial_dims=2,in_channels=32+64,out_channels=128,kernel_size=3,strides=1,padding=1,act='RELU',norm='BATCH'),

        ] # (B,104,H,W)->(B,H,H,W)
        self.lat_decoder_pipeline = nn.ModuleList(lat_decoder_list)        
        
        self.upconv2d_ap_pipeline = nn.ModuleList([
            nn.Identity(),
            Convolution(spatial_dims=2,act='RELU',in_channels=64,out_channels=64,is_transposed=True,kernel_size=3,strides=2,norm='BATCH'),])


        self.upconv2d_lat_pipeline = nn.ModuleList([
            nn.Identity(),
            Convolution(spatial_dims=2,act='RELU',in_channels=64,out_channels=64,is_transposed=True,kernel_size=3,strides=2,norm='BATCH'),])

        fusion_decoder_list = [
            Convolution(spatial_dims=3,in_channels=2,out_channels=32,strides=1,padding=1,kernel_size=3,act='RELU',norm='BATCH'),
            Convolution(spatial_dims=3,in_channels=32+2,out_channels=32,strides=1,padding=1,kernel_size=3,act='RELU',norm='BATCH')            
        ]

        self.fusion_decoder_pipeline = nn.ModuleList(fusion_decoder_list)

        self.upconv3d_pipeline = nn.ModuleList([nn.Identity(),Convolution(spatial_dims=3,act='RELU',in_channels=32,out_channels=32,is_transposed=True,kernel_size=3,strides=2,norm='BATCH')])

        self.segmentation_head = nn.Sequential(Convolution(spatial_dims=3,in_channels=32,out_channels=1,kernel_size=1,strides=1,padding=0,act=None,norm=None))

    def _encoder_layer(self):
        layers: List[nn.Module] = []
        # layers.append(nn.Conv2d(1,8,kernel_size=3,stride=1,padding=1)) # (B,1,H,W)->(B,8,H,W)
        layers.append(Convolution(spatial_dims=2,in_channels=1,out_channels=8,strides=1,padding=1,act='RELU',norm='BATCH'))
        return layers
    
    def forward(self, ap_image: torch.Tensor, lat_image: torch.Tensor):
        encoded_ap = self.ap_encoder(ap_image)
        encoded_lat = self.lat_encoder(lat_image)
        # [print('After 2D Encoding ',out.shape) for out in encoded_ap]

        dec_ap_fusion_out = []
        dec_lat_fusion_out = []
        dec_3d_fusion_out = []
        for i,(ap_decoder_2d,ap_upsampler,lat_decoder_2d,lat_upsampler,fuser,fuser_upsampler) in enumerate(zip(self.ap_decoder_pipeline,self.upconv2d_ap_pipeline,self.lat_decoder_pipeline,self.upconv2d_lat_pipeline,self.fusion_decoder_pipeline,self.upconv3d_pipeline)):
            if i == 0: # if this is the first in the sequence of decoders
                
                dec_ap = ap_decoder_2d(encoded_ap[i])
                dec_lat = lat_decoder_2d(encoded_lat[i])

                dec_ap_fusion_out.append(dec_ap)
                dec_lat_fusion_out.append(dec_lat)
            else: # other decoder takes input from adjacent encoder and previous decoder(upsampled to match HW)
                dec_ap = ap_decoder_2d(
                    torch.cat((ap_upsampler(dec_ap_fusion_out[i-1]),encoded_ap[i]),dim=1)
                )
                
                dec_lat = lat_decoder_2d(
                    torch.cat((lat_upsampler(dec_lat_fusion_out[i-1]),encoded_lat[i]),dim=1)
                )
                dec_ap_fusion_out.append(dec_ap)
                dec_lat_fusion_out.append(dec_lat)                

            if self.permute:
                # permute LAT decoded images (assume PIR orientation)
                permuted_dec_lat = dec_lat.swapdims(1,-1)
            else:
                permuted_dec_lat = dec_lat

            if i == 0:
                fused_out =  torch.cat((dec_ap.unsqueeze(dim=1),permuted_dec_lat.unsqueeze(dim=1)),dim=1)
                dec_3d_fusion_out.append(fuser(fused_out))

            else:
                fused_out =  torch.cat((fuser_upsampler(dec_3d_fusion_out[i-1]),dec_ap.unsqueeze(dim=1),permuted_dec_lat.unsqueeze(dim=1)),dim=1)
                dec_3d_fusion_out.append(fuser(fused_out))

        # [print('After 2D Decoding ',out.shape) for out in dec_ap_fusion_out]
        # [print('After 2D Decoding ',out.shape) for out in dec_lat_fusion_out]
        # [print('After 3D Decoding ',out.shape) for out in dec_3d_fusion_out]

        return self.segmentation_head(dec_3d_fusion_out[-1])

        # fused_out =  [torch.cat((ap.unsqueeze(dim=1),lat.unsqueeze(dim=1)),dim=1) 
        #     for ap,lat in zip(decoded_ap,permuted_decoded_lat)]

        # fusion_out = [ fusion_layer(fused_cube) for fused_cube,fusion_layer in zip(fused_out,self.fusion_decoder_pipeline)]
        # [print('Before UpSampling ',out.shape) for out in fusion_out]

        # fusion_scaled_out = [upsampler(out)for out,upsampler in zip(fusion_out,self.upconv3d_pipeline)]
        # [print('After UpSampling ',out.shape) for out in fusion_scaled_out]

        # return fusion_out
        # return decoded_ap,decoded_lat


if __name__ == '__main__':
    import torch 
    
    ap_img = torch.zeros((1,1,128,128))
    lat_img = torch.zeros((1,1,128,128))


    model = MultiScale2DPermuteConcat()
    fused_out = model(ap_img,lat_img)
    print(f'out {fused_out.shape}')
    # print(out_ap.shape,out_lat.shape)
    # for ap,lat in zip(out_ap,out_lat):
    #     print(ap.shape,lat.shape)
    # for out in fused_out:
    #     print(out.shape)