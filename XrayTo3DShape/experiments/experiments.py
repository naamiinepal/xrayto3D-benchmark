from .base_experiment import BaseExperiment
import torch
from typing import Tuple,Any

class VolumeAsInputExperiment(BaseExperiment):
    def __init__(self, model, optimizer, loss_function,batch_size, **kwargs: Any) -> None:
        super().__init__(model, optimizer, loss_function, batch_size,**kwargs)
    
    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']
        input = torch.cat((ap_tensor,lat_tensor),1)
        return [input],seg_tensor

class BiplanarAsInputExperiment(BaseExperiment):
    def __init__(self, model, optimizer, loss_function, batch_size, **kwargs: Any) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)
    
    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']
        input = (ap_tensor, lat_tensor)
        return input, seg_tensor
    