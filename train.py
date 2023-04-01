"""
script to train various architectures on given dataset
(defined by training and validation filepaths).
"""
import argparse
import os
import sys
from pathlib import Path

import monai.data.meta_obj as monai_meta_obj
import numpy as np
import pytorch_lightning as pl
import torch
from monai.utils.misc import set_determinism
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader

import wandb
import XrayTo3DShape
from XrayTo3DShape import (
    AutoencoderExperiment,
    BaseExperiment,
    CustomAutoEncoder,
    TLPredictorExperiment,
    anatomy_resolution_dict,
    get_anatomy_from_path,
    get_dataset,
    get_loss,
    get_model,
    get_model_config,
    get_transform_from_model_name,
    model_experiment_dict,
    printarr,
)

np.set_printoptions(precision=2, suppress=True)
torch.set_printoptions(precision=2, sci_mode=False)


def parse_training_arguments():
    """reads various options from commandline

    Returns:
        dict: (key,value) pairs of option and corresponding value
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("trainpaths")
    parser.add_argument("valpaths")
    parser.add_argument("--model_name")
    parser.add_argument("--loss")
    parser.add_argument("--size", type=int)
    parser.add_argument("--res", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)

    parser.add_argument("--epochs", type=int, default=-1)
    parser.add_argument("--steps", default=5000, type=int)

    parser.add_argument("--visualize", action="store_true", default=False)
    parser.add_argument("--debug", default=False, action="store_true")

    parser.add_argument("--tags", nargs="*")
    parser.add_argument("--wandb-project", default="2d-3d-benchmark")

    parser.add_argument("--lambda_bce", default=1.0)
    parser.add_argument("--lambda_dice", default=1.0)
    parser.add_argument("--make_sparse", default=False, action="store_true")

    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num_workers", default=os.cpu_count(), type=int)

    parser.add_argument("--dropout", default=False, action="store_true")
    parser.add_argument("--load_autoencoder_from", default="", type=str)
    parser.add_argument("--top_k_checkpoints", default=3, type=int)

    parser.add_argument("--precision", default=32, type=int)

    args = parser.parse_args()

    if args.precision == 16:
        args.precision = "bf16"
    return args


def update_args(args):
    """update sensible values for remaining arguments using
    the argument values provided. Perform sanity check of arguments.

    Args:
        args (dict): (option,value) pair
    """
    args.anatomy = get_anatomy_from_path(args.trainpaths)

    # add dropout to tag if exists
    if args.dropout:
        args.tags.append("dropout")

    # assert the resolution and size agree for each anatomy
    orig_size, orig_res = anatomy_resolution_dict[args.anatomy]
    assert int(args.size * args.res) == int(
        orig_size * orig_res
    ), f"({args.size},{args.res}) does not match ({orig_size},{orig_res})"
    args.experiment_name = model_experiment_dict[args.model_name]

    args.precision = 16 if args.gpu == 0 else 32  # use bfloat16 on RTX 3090


if __name__ == "__main__":
    args = parse_training_arguments()
    update_args(args)

    # print commandline arguments: all print outputs are logged by wandb
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
    model_name = args.model_name
    experiment_name = args.experiment_name
    WANDB_EXPERIMENT_GROUP = args.model_name
    WANDB_TAGS = [WANDB_EXPERIMENT_GROUP, ANATOMY, LOSS_NAME, *args.tags]

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)

    train_transforms = get_transform_from_model_name(
        model_name, image_size=IMG_SIZE, resolution=IMG_RESOLUTION
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

    model = get_model(
        model_name=args.model_name, image_size=IMG_SIZE, dropout=args.dropout
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
    if experiment_name == CustomAutoEncoder.__name__:
        experiment.make_sparse = args.make_sparse
    if experiment_name == TLPredictorExperiment.__name__:
        ae_model = get_model(model_name=CustomAutoEncoder.__name__, image_size=IMG_SIZE)
        if Path(args.load_autoencoder_from).exists():
            checkpoint = torch.load(args.load_autoencoder_from)
        else:
            raise ValueError(
                f"autoencoder checkpoint {args.load_autoencoder_from} does not exist"
            )
        for key in list(checkpoint["state_dict"].keys()):
            # model.layer1.conv1 -> layer1.conv1
            modified_key = key.replace("model.", "")
            value = checkpoint["state_dict"].pop(key)
            checkpoint["state_dict"][modified_key] = value
        if "loss_function.pos_weight" in checkpoint["state_dict"]:
            checkpoint["state_dict"].pop("loss_function.pos_weight")
        ae_model.load_state_dict(checkpoint["state_dict"])
        experiment.set_decoder(ae_model)  # type: ignore

    # run a sanity check
    batch = next(iter(train_loader))
    if args.experiment_name != AutoencoderExperiment.__name__:
        seg_meta_dict = experiment.get_segmentation_meta_dict(batch)
    batch_input, batch_output = experiment.get_input_output_from_batch(batch)

    pred_logits = experiment.model(*batch_input)
    if experiment_name == AutoencoderExperiment.__name__:
        pred_logits, latent_vec = pred_logits
    if experiment_name == TLPredictorExperiment.__name__:
        pred_logits = ae_model.latent_vec_decode(pred_logits)  # type: ignore
    loss = experiment.loss_function(pred_logits, batch_output).item()  # type: ignore
    input_zero = batch_input[0]
    printarr(pred_logits, batch_output, input_zero, loss)
    print(
        f"training samples {len(train_loader.dataset)} validation samples {len(val_loader.dataset)}"
    )
    print(f"Track meta data : {monai_meta_obj.get_track_meta()}")
    if args.debug:
        sys.exit()
    # loggers
    wandb_logger = WandbLogger(
        save_dir="runs/",
        project=WANDB_PROJECT,
        group=WANDB_EXPERIMENT_GROUP,
        tags=WANDB_TAGS,
    )
    wandb_logger.watch(model, log_graph=False)
    MODEL_CONFIG = get_model_config(model_name, IMG_SIZE)
    # save hyperparameters
    HYPERPARAMS = {
        "IMG_SIZE": IMG_SIZE,
        "RESOLUTION": IMG_RESOLUTION,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": lr,
        "SEED": SEED,
        "ANATOMY": ANATOMY,
        "MODEL_NAME": model_name,
        "LOSS": LOSS_NAME,
        "EXPERIMENT_NAME": experiment_name,
    }
    HYPERPARAMS.update(MODEL_CONFIG)
    wandb_logger.log_hyperparams(HYPERPARAMS)
    CHECKPOINT_FILENAME = (
        "epoch={epoch}-step={step}-val_loss={val/loss:.2f}"
        if experiment_name == CustomAutoEncoder.__name__
        else "epoch={epoch}-step={step}-val_dice={val/dice:.2f}"
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_last=True,
        save_top_k=args.top_k_checkpoints,
        filename=CHECKPOINT_FILENAME,
        auto_insert_metric_name=False,
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
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        enable_checkpointing=True,
        max_steps=args.steps,
    )

    trainer.fit(experiment, train_loader, val_loader)

    wandb.finish()
