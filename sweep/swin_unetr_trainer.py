"""SwinUNETR trainer"""

import argparse
import os
import sys

import pytorch_lightning as pl
from monai.utils.misc import set_determinism
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from monai.networks.nets.swin_unetr import SwinUNETR
import wandb
import XrayTo3DShape
from XrayTo3DShape import (
    BaseExperiment,
    anatomy_resolution_dict,
    get_anatomy_from_path,
    get_dataset,
    get_loss,
    get_transform_from_model_name,
    model_experiment_dict,
    printarr,
)


def parse_training_arguments():
    """parse arguments"""
    parser = argparse.ArgumentParser()
    # swinunetr specific arguments
    parser.add_argument("--num_heads", type=str)
    parser.add_argument("--feature_size", type=str)
    # other arguments
    parser.add_argument("--trainpaths")
    parser.add_argument("--valpaths")
    parser.add_argument("--loss")
    parser.add_argument("--size", type=int)
    parser.add_argument("--res", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--steps", default=5000, type=int)
    parser.add_argument("--wandb-project", default="swinunetr_sweep")
    parser.add_argument("--lambda_bce", default=1.0)
    parser.add_argument("--lambda_dice", default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num_workers", default=os.cpu_count(), type=int)
    parser.add_argument("--precision", default=32, type=int)
    parser.add_argument("--debug", default=False, action="store_true")
    args = parser.parse_args()

    if args.precision == 16:
        args.precision = "bf16"
    return args


def update_args(args):
    """update args"""
    args.anatomy = get_anatomy_from_path(args.trainpaths)

    # assert the resolution and size agree for each anatomy
    orig_size, orig_res = anatomy_resolution_dict[args.anatomy]
    assert int(args.size * args.res) == int(
        orig_size * orig_res
    ), f"({args.size},{args.res}) does not match ({orig_size},{orig_res})"
    args.experiment_name = model_experiment_dict[args.model_name]

    args.precision = 16 if args.gpu == 0 else 32  # use bfloat16 on RTX 3090

    if args.num_heads == "small":
        args.num_heads_val = (2, 2, 2, 2)
    if args.num_heads == "progressive":
        args.num_heads_val = (3, 6, 12, 24)

    if args.feature_size == 'small':
        args.feature_size_val = 12
    if args.feature_size == 'default':
        args.feature_size_val = 24
    if args.feature_size == 'large':
        args.feature_size_val = 48
        
if __name__ == "__main__":
    args = parse_training_arguments()
    args.model_name = SwinUNETR.__name__
    update_args(args)

    print(sys.argv)
    print(args)

    SEED = 12345
    lr = args.lr
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.size
    ANATOMY = args.anatomy
    LOSS_NAME = args.loss
    IMG_RESOLUTION = args.res
    BATCH_SIZE = args.batch_size
    WANDB_PROJECT = args.wandb_project
    MODEL_NAME = SwinUNETR.__name__
    experiment_name = args.experiment_name
    WANDB_EXPERIMENT_GROUP = MODEL_NAME
    WANDB_TAGS = [WANDB_EXPERIMENT_GROUP, ANATOMY, LOSS_NAME]

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)

    train_transforms = get_transform_from_model_name(
        MODEL_NAME, image_size=IMG_SIZE, resolution=IMG_RESOLUTION
    )

    train_loader = DataLoader(
        get_dataset(args.trainpaths, transforms=train_transforms),
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        get_dataset(args.valpaths, transforms=train_transforms),
        batch_size=BATCH_SIZE,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )

    model = SwinUNETR(
        img_size=IMG_SIZE,
        in_channels=2,
        out_channels=1,
        num_heads=args.num_heads_val,
        feature_size=args.feature_size_val,
        drop_rate=0.1,
        spatial_dims=3,
    )

    loss_function = get_loss(
        loss_name=LOSS_NAME,
        anatomy=ANATOMY,
        image_size=IMG_SIZE,
        lambda_bce=args.lambda_bce,
        lambda_dice=args.lambda_dice,
        device=f"cuda:{args.gpu}",
    )
    optimizer = Adam(model.parameters(), lr)
    # load pytorch lightning module
    experiment: BaseExperiment = getattr(XrayTo3DShape.experiments, experiment_name)(
        model, optimizer, loss_function, BATCH_SIZE
    )
    # run a sanity check
    batch = next(iter(train_loader))
    batch_input, batch_output = experiment.get_input_output_from_batch(batch)
    pred_logits = experiment.model(*batch_input)
    loss = experiment.loss_function(pred_logits, batch_output).item()  # type: ignore
    input_zero = batch_input[0]
    printarr(pred_logits, batch_output, input_zero, loss)
    print(
        f"training samples {len(train_loader.dataset)} validation samples {len(val_loader.dataset)}"
    )
    if args.debug:
        sys.exit()
    # loggers
    wandb_logger = WandbLogger(
        save_dir="runs/",
        project=WANDB_PROJECT,
        group=WANDB_EXPERIMENT_GROUP,
        tags=WANDB_TAGS,
    )
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        max_epochs=NUM_EPOCHS,
        devices=[args.gpu],
        deterministic=False,
        log_every_n_steps=1,
        auto_select_gpus=True,
        logger=[wandb_logger],
        enable_progress_bar=True,
        enable_checkpointing=False,
        max_steps=args.steps,
    )

    trainer.fit(experiment, train_loader, val_loader)

    wandb.finish()
