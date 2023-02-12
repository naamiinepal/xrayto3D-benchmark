import torch
from torch.nn import BCEWithLogitsLoss
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from XrayTo3DShape import get_dataset,get_kasten_transforms,VolumeAsInputExperiment,NiftiPredictionWriter,parse_training_arguments
from monai.utils.misc import set_determinism
from monai.networks.nets.unet import UNet
from monai.losses.dice import DiceLoss,DiceCELoss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything


if __name__ == '__main__':
    

    args = parse_training_arguments()

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
    model_config = {
        "in_channels": 2,
        "out_channels": 1,
        "channels": (64, 128, 256, 512),
        "strides": (2,2,2,2),
        "act": "RELU",
        "norm": "BATCH",
        "num_res_units": 2,
        "dropout": 0.1,
    }

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)  

    train_transforms = get_kasten_transforms(size=IMG_SIZE,resolution=IMG_RESOLUTION)
    train_loader = DataLoader(get_dataset(args.trainpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20,shuffle=True)
    val_loader = DataLoader(get_dataset(args.valpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20,shuffle=False)



    model = UNet(spatial_dims=3,**model_config)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = Adam(model.parameters(), lr)

    experiment = VolumeAsInputExperiment(model,optimizer,loss_function,BATCH_SIZE)
    if args.evaluate and args.save_predictions:
        nifti_saver = NiftiPredictionWriter(output_dir=args.output_dir,write_interval='batch')
        trainer = pl.Trainer(callbacks=[nifti_saver])
        trainer.predict(model=experiment,ckpt_path=args.checkpoint_path,dataloaders=val_loader,return_predictions=False)
    else:
        # loggers
        wandb_logger = WandbLogger(save_dir='runs/',project=WANDB_PROJECT,group=EXPERIMENT,tags=[EXPERIMENT,'model_selection',*args.tags])
        wandb_logger.log_hyperparams({'model':model_config})
        trainer = pl.Trainer(accelerator=args.accelerator,precision=args.precision,max_epochs=-1,gpus=[args.gpu],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[wandb_logger],enable_progress_bar=True,enable_checkpointing=True)
        
        trainer.fit(experiment,train_loader,val_loader)