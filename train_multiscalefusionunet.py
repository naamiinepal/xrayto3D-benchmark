import torch
from XrayTo3DShape import BaseDataset,get_nonkasten_transforms,TwoDPermuteConcatMultiScale
from torch.utils.data import DataLoader,Dataset
from monai.losses.dice import DiceCELoss,DiceLoss
from monai.metrics.meandice import DiceMetric,compute_dice
from monai.transforms import *
from monai.metrics.metric import CumulativeIterationMetric,Cumulative
from monai.utils.misc import set_determinism
import argparse
from typing import Callable,Dict
import pytorch_lightning as pl
from pytorch_lightning import LightningModule,seed_everything
from pytorch_lightning.loggers import WandbLogger,TensorBoardLogger
import warnings
warnings.filterwarnings("ignore")

def get_dataset(filepaths:str,transforms:Dict)->Dataset:
    import pandas as pd
    paths = pd.read_csv(filepaths,index_col=0).to_numpy()
    paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]
    ds = BaseDataset(data=paths, transforms=transforms)
    return ds

def train(model:torch.nn.Module,optimizer:torch.optim.Optimizer,loss_function:Callable,trainloader:DataLoader,valloader:DataLoader,metric_evaluator:CumulativeIterationMetric ):
    epochwise_loss_aggregator = Cumulative()
    for batch in trainloader:
        # setup input output pairs
        ap,lat,seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]
        input_volume = torch.cat((ap_tensor,lat_tensor),1)
        
        optimizer.zero_grad()
        pred_logits = model(input_volume)
        loss = loss_function(pred_logits,seg_tensor)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            eval_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
            pred = eval_transform(pred_logits)
            metric_evaluator(y_pred=pred, y=seg_tensor)
            epochwise_loss_aggregator.append(loss)

        # print(f'loss {loss.item():.4f}')
    aggregate_loss = torch.mean(epochwise_loss_aggregator.get_buffer()) # type: ignore    
    print(f'Loss {aggregate_loss.item():.4f} Dice score {metric_evaluator.aggregate().item():.4f}')
    metric_evaluator.reset()
    epochwise_loss_aggregator.reset()

class MultiScaleFusionUNet_XrayTo3D(LightningModule):
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
        pred_logits = model(ap_tensor,lat_tensor)
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
        pred_logits = model(ap_tensor,lat_tensor)
        pred = self.val_transform(pred_logits)
        val_loss = self.loss_function(pred_logits,seg_tensor)
        dice_metric = torch.mean(compute_dice(pred,seg_tensor))
        
        self.log('val_dice',dice_metric.item(),on_step=False,on_epoch=True,prog_bar=True,batch_size=BATCH_SIZE)
        self.log('val_loss',val_loss,on_step=False,on_epoch=True,prog_bar=True,batch_size=BATCH_SIZE)
        return {"pred":pred}

    def configure_optimizers(self):
        return self.optimizer

if __name__ == '__main__':
    SEED = 12345
    EXPERIMENT = 'MultiScaleFusionUNet'
    lr = 1e-2
    NUM_EPOCHS = 100
    IMG_SIZE = 64
    IMG_RESOLUTION = 1.5
    WANDB_ON = False
    TEST_ZERO_INPUT = False
    BATCH_SIZE = 4
    WANDB_PROJECT = 'pipeline-test-01'
    config_multiscalefusionunet = {
        "in_shape": (1,IMG_SIZE,IMG_SIZE),
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

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)    
    parser = argparse.ArgumentParser()
    parser.add_argument('trainpaths')
    parser.add_argument('valpaths')

    args = parser.parse_args()
    train_transforms = get_nonkasten_transforms()
    train_loader = DataLoader(get_dataset(args.trainpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20)
    val_loader = DataLoader(get_dataset(args.valpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20)

    # loggers
    wandb_logger = WandbLogger(save_dir='runs/',project=WANDB_PROJECT,group=EXPERIMENT,tags=[EXPERIMENT,'model_selection'])
    wandb_logger.log_hyperparams({'model':config_multiscalefusionunet})
    tensorboard_logger = TensorBoardLogger(save_dir='runs/lightning_logs',name=EXPERIMENT)
    tensorboard_logger.log_hyperparams({'model':config_multiscalefusionunet})

    model = TwoDPermuteConcatMultiScale(config_multiscalefusionunet)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    dice_metric_evaluator = DiceMetric(include_background=False)

    attunet_experiment = MultiScaleFusionUNet_XrayTo3D(model,optimizer,loss_function)
    trainer = pl.Trainer(accelerator='gpu',max_epochs=-1,gpus=[0],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[tensorboard_logger,wandb_logger],enable_progress_bar=True,enable_checkpointing=True)
    trainer.fit(attunet_experiment,train_loader,val_loader)