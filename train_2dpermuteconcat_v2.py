import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from XrayTo3DShape import get_dataset,get_nonkasten_transforms,BiplanarAsInputExperiment,NiftiPredictionWriter,parse_training_arguments,TwoDPermuteConcatModel
from monai.utils.misc import set_determinism
from monai.losses.dice import DiceLoss
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything


if __name__ == '__main__':
    

    args = parse_training_arguments()
    SEED = 12345
    EXPERIMENT = 'TwoDPermuteConcat'
    lr = 1e-2
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.size
    IMG_RESOLUTION = args.res
    WANDB_ON = False
    TEST_ZERO_INPUT = False
    BATCH_SIZE = args.batch_size
    WANDB_PROJECT = 'pipeline-test-01'
    model_config_64 = {
        "input_image_size": [args.size, args.size],
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
  
    model_config_128= {
        "input_image_size": [args.size, args.size],
        "encoder": {
            "in_channels": [1, 16, 32, 32, 32, 32],
            "out_channels": [16, 32, 32, 32, 32, 32],
            "strides": [2, 2, 1, 1, 1, 1],
            "kernel_size": 7,
        },
        "ap_expansion": {
            "in_channels": [32, 32, 32, 32,32],
            "out_channels": [32, 32, 32, 32, 32],
            "strides": ((2, 1, 1),) * 5,
            "kernel_size": 3,
        },
        "lat_expansion": {
            "in_channels": [32, 32, 32, 32, 32],
            "out_channels": [32, 32, 32, 32, 32],
            "strides": ((1, 1, 2),) * 5,
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

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)  

    train_transforms = get_nonkasten_transforms(size=IMG_SIZE,resolution=IMG_RESOLUTION)
    train_loader = DataLoader(get_dataset(args.trainpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20,shuffle=True,drop_last=False)
    val_loader = DataLoader(get_dataset(args.valpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20,shuffle=False,drop_last=True)


    model_config = model_config_64 if IMG_SIZE == 64 else model_config_128
    model = TwoDPermuteConcatModel(model_config)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    experiment = BiplanarAsInputExperiment(model,optimizer,loss_function,BATCH_SIZE)
    if args.evaluate and args.save_predictions:
        nifti_saver = NiftiPredictionWriter(output_dir=args.output_dir,write_interval='batch')
        trainer = pl.Trainer(callbacks=[nifti_saver])
        trainer.predict(model=experiment,ckpt_path=args.checkpoint_path,dataloaders=val_loader,return_predictions=False)
    else:
        # loggers
        wandb_logger = WandbLogger(save_dir='runs/',project=WANDB_PROJECT,group=EXPERIMENT,tags=[EXPERIMENT,'model_selection',*args.tags])
        wandb_logger.log_hyperparams({'model':model_config_64})
        trainer = pl.Trainer(accelerator=args.accelerator,precision=args.precision,max_epochs=-1,gpus=[args.gpu],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[wandb_logger],enable_progress_bar=True,enable_checkpointing=True)
        
        trainer.fit(experiment,train_loader,val_loader)