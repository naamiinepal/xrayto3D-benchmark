import pytorch_lightning as pl
from typing  import Any,Tuple, Optional
from XrayTo3DShape import post_transform
import torch
from monai.metrics.meandice import compute_dice

class BaseExperiment(pl.LightningModule):
    def __init__(self, model, optimizer,loss_function,batch_size, **kwargs: Any) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size

    def get_input_output_from_batch(self,batch)->Tuple[Any,torch.Tensor]:
        raise NotImplementedError()

    def get_segmentation_meta_dict(self, batch):
        ap, lat, seg = batch
        return seg['seg_meta_dict']

    def training_step(self, batch, batch_idx):
        input, output = self.get_input_output_from_batch(batch)
        pred_logits = self.model(*input)
        loss = self.loss_function(pred_logits,output)
        with torch.no_grad():
            pred = post_transform(pred_logits.detach())
            dice_metric = torch.mean(compute_dice(pred,output))
        
        self.log('train/loss',loss.item(),on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log('train/dice',dice_metric.item(), on_step=True,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        return loss
    
    def validation_step(self,batch,batch_idx):
        input,output = self.get_input_output_from_batch(batch)
        pred_logits = self.model(*input)
        loss = self.loss_function(pred_logits,output)
        pred = post_transform(pred_logits)
        dice_metric = torch.mean(compute_dice(pred,output,ignore_empty=False))
        self.log('val/loss',loss.item(),on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)
        self.log('val/dice',dice_metric.item(),on_step=False,on_epoch=True,prog_bar=True,batch_size=self.batch_size)

    
    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        input,output = self.get_input_output_from_batch(batch)
        pred_logits = self.model(*input)
        pred = post_transform(pred_logits)

        out = {}
        seg_meta_dict = self.get_segmentation_meta_dict(batch)

        out['pred'] = pred
        out['gt'] = output
        out['seg_meta_dict'] = seg_meta_dict

        return out
    
    def configure_optimizers(self):
        return self.optimizer