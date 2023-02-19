import wandb
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
import pytorch_lightning as pl
from XrayTo3DShape import (
    get_dataset,
    get_nonkasten_transforms,
    get_kasten_transforms,
    VolumeAsInputExperiment,
    BiplanarAsInputExperiment,
    NiftiPredictionWriter,
    parse_training_arguments,
    get_model,
    get_model_config
)
import XrayTo3DShape
from monai.utils.misc import set_determinism
from monai.losses.dice import DiceLoss
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning import seed_everything


if __name__ == "__main__":

    args = parse_training_arguments()
    SEED = 12345
    lr = 1e-2
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.size
    IMG_RESOLUTION = args.res
    BATCH_SIZE = args.batch_size
    WANDB_PROJECT = "pipeline-test-01"
    model_name = args.model_name
    experiment_name = args.experiment_name
    WANDB_EXPERIMENT_GROUP = args.model_name
    WANDB_TAGS = [WANDB_EXPERIMENT_GROUP,'model_selection',*args.tags]

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)

    if experiment_name == BiplanarAsInputExperiment.__name__:
        callable_transform = get_nonkasten_transforms
    elif experiment_name == VolumeAsInputExperiment.__name__:
        callable_transform = get_kasten_transforms
    else:
        raise ValueError(f'Invalid experiment name {experiment_name}')
    train_transforms = callable_transform(size=IMG_SIZE,resolution=IMG_RESOLUTION)

    
    train_loader = DataLoader(
        get_dataset(args.trainpaths, transforms=train_transforms),
        batch_size=BATCH_SIZE,
        num_workers=20,
        shuffle=True,
    )
    val_loader = DataLoader(
        get_dataset(args.valpaths, transforms=train_transforms),
        batch_size=BATCH_SIZE,
        num_workers=20,
        shuffle=False,
    )

    model = get_model(model_name=args.model_name,image_size=IMG_SIZE)
    MODEL_CONFIG = get_model_config(model_name,IMG_SIZE)

    loss_function = DiceLoss(sigmoid=True)
    optimizer = Adam(model.parameters(), lr)
    # load pytorch lightning module
    experiment = getattr(XrayTo3DShape.experiments,experiment_name)(model,optimizer,loss_function,BATCH_SIZE)

    if args.evaluate and args.save_predictions:
        nifti_saver = NiftiPredictionWriter(output_dir=args.output_dir,write_interval='batch')
        trainer = pl.Trainer(callbacks=[nifti_saver])
        trainer.predict(model=experiment,ckpt_path=args.checkpoint_path,dataloaders=val_loader,return_predictions=False)
    else:
        # loggers
        wandb_logger = WandbLogger(save_dir='runs/',project=WANDB_PROJECT,group=WANDB_EXPERIMENT_GROUP,tags=WANDB_TAGS)
        wandb_logger.log_hyperparams({'model':MODEL_CONFIG})

        trainer = pl.Trainer(accelerator=args.accelerator,precision=args.precision,max_epochs=3,devices=[args.gpu],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[wandb_logger],enable_progress_bar=True,enable_checkpointing=True)

        trainer.fit(experiment,train_loader,val_loader)


        wandb.finish()