from .base_experiment import BaseExperiment
import torch
from typing import Tuple,Any
import wandb
from ..architectures import CustomAutoEncoder
from ..utils import reproject,to_numpy
from ..transforms import post_transform
from monai.metrics.meandice import compute_dice
from typing import Optional

class VolumeAsInputExperiment(BaseExperiment):
    def __init__(self, model, optimizer, loss_function,batch_size, **kwargs: Any) -> None:
        super().__init__(model, optimizer, loss_function, batch_size,**kwargs)

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']
        input = torch.cat((ap_tensor,lat_tensor),1)
        return [input],seg_tensor

class SingleHeadExperiment(BaseExperiment): # this is a work in progress. Created for SwinUNETR, but is not required now. 
    def __init__(self, model, optimizer, loss_function, batch_size, **kwargs: Any) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)
    
    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']
        input = torch.cat((ap_tensor,lat_tensor),dim=1)
        return [input], seg_tensor # the input has to be put inside a list as a hack to keep the interface same for all type of models self.model(*input)

class ParallelHeadsExperiment(BaseExperiment):
    def __init__(self, model, optimizer, loss_function, batch_size, **kwargs: Any) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)
    
    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']
        input = (ap_tensor, lat_tensor)
        return input, seg_tensor
    

class AutoencoderExperiment(BaseExperiment):
    def __init__(self, model, optimizer, loss_function, batch_size, 
                 make_sparse=False, **kwargs: Any) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)
        self.SPARSITY_REG_STRENGTH = 1e-3
        self.make_sparse = make_sparse

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        noisy_seg, clean_seg  = seg['gaus'],seg['orig']
        return [noisy_seg],clean_seg # the input has to be put inside a list as a hack to keep the interface same for all type of models self.model(*input)
    
    def sparse_loss(self,latent_vector:torch.Tensor):
        return torch.mean(torch.abs(latent_vector))
    
    def training_step(self, batch, batch_idx):
        input,output = self.get_input_output_from_batch(batch)
        pred_logits, latent_vector = self.model(*input)
        supervised_loss = self.loss_function(pred_logits,output)
        if self.make_sparse:
            sparsity_loss = self.sparse_loss(latent_vector)
            total_loss = supervised_loss + self.SPARSITY_REG_STRENGTH * sparsity_loss
        else:
            total_loss = supervised_loss
        self.log('train/loss',total_loss.item(),on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        if self.global_step % 20 and batch_idx == 0:
            with torch.no_grad():
                self.log_3d_images(pred_logits.detach(),label='train/predictions')
                self.log_3d_images(output,label='train/groundtruth')
                pass
        return total_loss

    def validation_step(self, batch, batch_idx):
        input,output = self.get_input_output_from_batch(batch)
        pred_logits, latent_vector = self.model(*input)
        supervised_loss = self.loss_function(pred_logits,output)
        if self.make_sparse:
            sparsity_loss = self.sparse_loss(latent_vector)
            total_loss = supervised_loss + self.SPARSITY_REG_STRENGTH * sparsity_loss
        else:
            total_loss = supervised_loss
        self.log('val/loss',total_loss.item(),on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        if self.global_step % 20 and batch_idx == 0:
            self.log_3d_images(pred_logits.detach(),label='val/predictions')
            self.log_3d_images(output,label='val/groundtruth')

            pass


    def log_3d_images(self, predictions,label):
        projections = [
            reproject(volume.squeeze(), 0) for volume in predictions]
        wandb.log({f"{label}": [wandb.Image(
            to_numpy(image.float())) for image in projections]})

class TLPredictorExperiment(BaseExperiment):
    def __init__(self, model, optimizer, loss_function, batch_size, 
                                **kwargs: Any) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)
        self.TNet: CustomAutoEncoder

    def set_decoder(self,autoencoder_model:CustomAutoEncoder):
        self.TNet = autoencoder_model
        self.TNet.eval() # set to eval mode        

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']
        input = torch.cat((ap_tensor,lat_tensor),1)
        return [input],seg_tensor

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        input,output = self.get_input_output_from_batch(batch)
        pred_latent_vec = self.model(*input)
        pred_logits = self.TNet.latent_vec_decode(pred_latent_vec)
        pred = post_transform(pred_logits)

        out = {}
        seg_meta_dict = self.get_segmentation_meta_dict(batch)

        out['pred'] = pred
        out['gt'] = output
        out['seg_meta_dict'] = seg_meta_dict

        return out
    
    def training_step(self, batch, batch_idx):
        input, output = self.get_input_output_from_batch(batch)
        pred_latent_vec = self.model(*input)
       
        pred_logits = self.TNet.latent_vec_decode(pred_latent_vec)
        loss = self.loss_function(pred_logits,output)
        with torch.no_grad():
            pred = post_transform(pred_logits.detach())
            dice_metric = torch.mean(compute_dice(pred,output))
        
        self.log('train/loss',loss.item(),on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log('train/dice',dice_metric.item(), on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input, output = self.get_input_output_from_batch(batch)
        pred_latent_vec = self.model(*input)
        pred_logits = self.TNet.latent_vec_decode(pred_latent_vec)
        loss = self.loss_function(pred_logits,output)
        pred = post_transform(pred_logits.detach())
        dice_metric = torch.mean(compute_dice(pred,output,ignore_empty=False))
        self.log('val/loss',loss.item(),on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log('val/dice',dice_metric.item(),on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
