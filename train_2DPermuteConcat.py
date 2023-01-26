import torch
from XrayTo3DShape import get_dataset,get_nonkasten_transforms,TwoDPermuteConcat
from torch.utils.data import DataLoader,Dataset
from monai.losses.dice import DiceCELoss
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


class TwoDPermuteConcat_XrayTo3D(LightningModule):
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
    EXPERIMENT = 'TwoDPermuteConcat'
    lr = 1e-2
    NUM_EPOCHS = 100
    IMG_SIZE = 64
    IMG_RESOLUTION = 1.5
    WANDB_ON = False
    TEST_ZERO_INPUT = False
    BATCH_SIZE = 4
    WANDB_PROJECT = 'pipeline-test-01'
    config_bayat = {
        "input_image_size": [64, 64],
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
        "decoder": {
            "in_channels": [64, 64, 64, 64, 64, 32, 16],
            "out_channels": [64, 64, 64, 64, 32, 16, 1],
            "strides": (1, 1, 1, 1, 2, 2, 1),
            "kernel_size": (3,3,3,3,3,3,7),
        },
        "act": "RELU",
        "norm": "BATCH",
        "dropout": 0.0,
        "bias": False
    }
    model = TwoDPermuteConcat(config_bayat)

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
    wandb_logger.log_hyperparams({'model':config_bayat})
    tensorboard_logger = TensorBoardLogger(save_dir='runs/lightning_logs',name=EXPERIMENT)
    tensorboard_logger.log_hyperparams({'model':config_bayat})


    model = TwoDPermuteConcat(config_bayat)
    loss_function = DiceCELoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    dice_metric_evaluator = DiceMetric(include_background=False)

    attunet_experiment = TwoDPermuteConcat_XrayTo3D(model,optimizer,loss_function)
    trainer = pl.Trainer(accelerator='gpu',max_epochs=-1,gpus=[0],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[tensorboard_logger,wandb_logger],enable_progress_bar=True,enable_checkpointing=True)
    trainer.fit(attunet_experiment,train_loader,val_loader)