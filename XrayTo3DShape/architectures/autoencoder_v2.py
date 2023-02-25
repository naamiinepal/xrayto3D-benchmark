from torch import nn
from typing import Dict,Sequence, Union,Optional,Tuple
from monai.networks.blocks.convolutions import Convolution
from monai.networks.nets.autoencoder import AutoEncoder
import numpy as np
import math
from functools import reduce
import operator

class CustomAutoEncoder(AutoEncoder):
    def __init__(self,latent_dim:int, spatial_dims: int, in_channels: int, out_channels: int, channels: Sequence[int], strides: Sequence[int], kernel_size: Union[Sequence[int], int] = 3, up_kernel_size: Union[Sequence[int], int] = 3, num_res_units: int = 0, inter_channels: Optional[list] = None, inter_dilations: Optional[list] = None, num_inter_units: int = 2, act: Optional[Union[Tuple, str]] = 'RELU', norm: Union[Tuple, str] = 'INSTANCE', dropout: Optional[Union[Tuple, str, float]] = None, bias: bool = True) -> None:
        super().__init__(spatial_dims, in_channels, out_channels, channels, strides, kernel_size, up_kernel_size, num_res_units, inter_channels, inter_dilations, num_inter_units, act, norm, dropout, bias)

        self.bottleneck_fcn_encode = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod((channels[-1],2,2,2)),latent_dim)
        )

        self.bottleneck_fcn_decode = nn.Sequential(
            nn.Linear(latent_dim,np.prod((channels[-1],2,2,2)))
            
        )

        self._initialize_weights()

    def latent_vec_decode(self, latent_vec):
        x = self.bottleneck_fcn_decode(latent_vec)
        x = x.reshape(x.shape[0],-1,2,2,2)
        pred_logits = self.decode(x) 
        return pred_logits
    
    def forward(self, x):
        x = self.encode(x)
        latent_vec = self.bottleneck_fcn_encode(x)
        x = self.latent_vec_decode(latent_vec)
        return x, latent_vec

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

class TLPredictor(nn.Module):
    def __init__(self,
                spatial_dims: int,
                in_channel: int,
                latent_dim : int,
                kernel_size: Union[Sequence[int],int] = 3,
                act: Optional[Union[Tuple, str]] = "RELU",
                norm: Union[Tuple, str] = "INSTANCE",
                dropout: Optional[Union[Tuple, str, float]] = None,
                bias: bool = True,                 
                 ):
        super().__init__()
        self.spatial_dims = spatial_dims


        self.model = nn.Sequential(
            Convolution(spatial_dims=3,in_channels=in_channel,out_channels=16,strides=2,kernel_size=kernel_size,act=act,norm=norm,bias=bias,dropout=dropout),
            Convolution(spatial_dims=3,in_channels=16,out_channels=32,strides=2,kernel_size=kernel_size,act=act,norm=norm,bias=bias,dropout=dropout),
            Convolution(spatial_dims=3,in_channels=32,out_channels=64,strides=2,kernel_size=kernel_size,act=act,norm=norm,bias=bias,dropout=dropout),  
            Convolution(spatial_dims=3,in_channels=64,out_channels=128,strides=2,kernel_size=kernel_size,act=act,norm=norm,bias=bias,dropout=dropout),                      
            nn.Flatten(),
            nn.Linear(np.prod((128, 8,8,8,)),latent_dim)
        )
        
    def forward(self,x):
        return self.model(x)


