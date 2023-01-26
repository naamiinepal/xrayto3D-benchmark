import torch
from XrayTo3DShape import get_dataset,get_kasten_transforms
from torch.utils.data import DataLoader,Dataset
from monai.losses.dice import DiceLoss
from monai.metrics.meandice import DiceMetric,compute_dice
from monai.networks.nets.unet import UNet
from monai.transforms import *
from monai.metrics.metric import CumulativeIterationMetric,Cumulative
from monai.utils.misc import set_determinism
import argparse
from typing import Callable,Dict,Optional,Any,Sequence
import pytorch_lightning as pl
from pytorch_lightning import LightningModule,seed_everything
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
from XrayTo3DShape import NiftiPredictionWriter
import numpy as np

import warnings
warnings.filterwarnings("ignore")


class UNet_XrayTo3D(LightningModule):
    def __init__(self,model,optimizer:torch.optim.Optimizer,loss_function:Callable) -> None:
        super().__init__()
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.val_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])


    def training_step(self,batch,batch_idx):
        # setup input output pairs
        ap,lat,seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]
        input_volume = torch.cat((ap_tensor,lat_tensor),1)
        pred_logits = model(input_volume)
        loss = self.loss_function(pred_logits,seg_tensor)
        with torch.no_grad():
            pred = self.val_transform(pred_logits.detach())
            dice_metric = torch.mean(compute_dice(pred,seg_tensor))
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True,batch_size=BATCH_SIZE)  
        self.log("train_dice",dice_metric.item(),on_step=True,on_epoch=True,prog_bar=True,batch_size=BATCH_SIZE)      
        return loss

    def validation_step(self,batch,batch_idx):
        # setup input output pairs
        ap,lat,seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]
        input_volume = torch.cat((ap_tensor,lat_tensor),1)
        pred_logits = model(input_volume)
        pred = self.val_transform(pred_logits)
        val_loss = self.loss_function(pred_logits,seg_tensor)
        dice_metric = torch.mean(compute_dice(pred,seg_tensor))
        
        self.log('val_dice',dice_metric.item(),on_step=False,on_epoch=True,prog_bar=True,batch_size=BATCH_SIZE)
        self.log('val_loss',val_loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=BATCH_SIZE)
        return {"pred":pred}

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        ap,lat,seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]
        input_volume = torch.cat((ap_tensor,lat_tensor),1)
        pred_logits = model(input_volume)
        pred = self.val_transform(pred_logits)

        pred_seg = {}
        pred_seg['pred_seg'] = pred
        pred_seg['pred_seg_meta_dict'] = seg['seg_meta_dict']
        return pred_seg

    def configure_optimizers(self):
        return self.optimizer

if __name__ == '__main__':
  
    parser = argparse.ArgumentParser()
    parser.add_argument('trainpaths')
    parser.add_argument('valpaths')
    parser.add_argument('--tags',nargs='*')
    parser.add_argument('--gpu',type=int,default=1)
    parser.add_argument('--accelerator',default='gpu')
    parser.add_argument('--size',type=int,default=64)
    parser.add_argument('--res',type=float,default=1.5)
    parser.add_argument('--batch_size',type=int,default=4)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--evaluate',default=False,action='store_true')
    parser.add_argument('--save_predictions',default=False,action='store_true')
    parser.add_argument('--checkpoint_path')
    parser.add_argument('--output_dir')
    parser.add_argument('--precision',default='32',type=str)

    args = parser.parse_args()

    SEED = 12345
    EXPERIMENT = 'Unet'
    lr = 1e-2
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.size
    IMG_RESOLUTION = args.res
    WANDB_ON = False
    TEST_ZERO_INPUT = False
    BATCH_SIZE = args.batch_size
    WANDB_PROJECT = 'pipeline-test-01'
    config_kasten = {
        "in_channels": 2,
        "out_channels": 1,
        "channels": (64, 128, 256,512),
        "strides": (2,2,2,2),
        "act": "RELU",
        "norm": "BATCH",
        "num_res_units": 2,
    }

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)  

    train_transforms = get_kasten_transforms(size=IMG_SIZE,resolution=IMG_RESOLUTION)
    train_loader = DataLoader(get_dataset(args.trainpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20,shuffle=True)
    val_loader = DataLoader(get_dataset(args.valpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20,shuffle=False)



    model = UNet(spatial_dims=3,**config_kasten)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    dice_metric_evaluator = DiceMetric(include_background=False)

    attunet_experiment = UNet_XrayTo3D(model,optimizer,loss_function)
    if args.evaluate and args.save_predictions:
        nifti_saver = NiftiPredictionWriter(output_dir=args.output_dir,write_interval='batch')
        trainer = pl.Trainer(callbacks=[nifti_saver])
        trainer.predict(model=attunet_experiment,ckpt_path=args.checkpoint_path,dataloaders=val_loader,return_predictions=False)
    else:
        # loggers
        wandb_logger = WandbLogger(save_dir='runs/',project=WANDB_PROJECT,group=EXPERIMENT,tags=[EXPERIMENT,'model_selection',*args.tags])
        wandb_logger.log_hyperparams({'model':config_kasten})
        tensorboard_logger = TensorBoardLogger(save_dir='runs/lightning_logs',name=EXPERIMENT)
        tensorboard_logger.log_hyperparams({'model':config_kasten})
        trainer = pl.Trainer(accelerator=args.accelerator,precision=args.precision,max_epochs=-1,gpus=[args.gpu],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[tensorboard_logger,wandb_logger],enable_progress_bar=True,enable_checkpointing=True)
        
        trainer.fit(attunet_experiment,train_loader,val_loader)