import torch
from XrayTo3DShape import get_dataset,get_nonkasten_transforms,TwoDPermuteConcatMultiScale,BiplanarAsInputExperiment,parse_training_arguments,NiftiPredictionWriter
from torch.utils.data import DataLoader
from monai.losses.dice import DiceCELoss,DiceLoss
from monai.utils.misc import set_determinism
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    args = parse_training_arguments()

    SEED = 12345
    EXPERIMENT = 'MultiScaleFusionUNet'
    lr = 1e-2
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.size
    IMG_RESOLUTION = args.res
    WANDB_ON = False
    TEST_ZERO_INPUT = False
    BATCH_SIZE = args.batch_size
    WANDB_PROJECT = 'pipeline-test-01'
    model_config = {
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

    train_transforms = get_nonkasten_transforms()
    train_loader = DataLoader(get_dataset(args.trainpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20)
    val_loader = DataLoader(get_dataset(args.valpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20)



    model = TwoDPermuteConcatMultiScale(model_config)
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
        wandb_logger.log_hyperparams({'model':model_config})
        trainer = pl.Trainer(accelerator=args.accelerator,precision=args.precision,max_epochs=-1,gpus=[args.gpu],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[wandb_logger],enable_progress_bar=True,enable_checkpointing=True)
        
        trainer.fit(experiment,train_loader,val_loader)