"""
run inference using model checkpoint.
save metrics to csv log, predictions as nifti
"""
import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import XrayTo3DShape
from XrayTo3DShape import (
    MetricsLogger,
    NiftiPredictionWriter,
    get_dataset,
    get_latest_checkpoint,
    get_model,
    get_transform_from_model_name,
    model_experiment_dict,
)


def parse_evaluation_arguments():
    """read options for running inference from
    model checkpoint.

    Returns:
        dict: (option,value)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--testpaths")
    parser.add_argument("--model_name")
    parser.add_argument("--ckpt_path")
    parser.add_argument("--res", type=float)
    parser.add_argument("--nsd_tolerance", type=float, default=1.5)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--num_workers", default=20, type=int)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default=32)
    return parser.parse_args()


def update_args(args):
    """infer/fill-in reasonable defaults from
    partial arguments

    Args:
        args (dict): (key,value)

    Returns:
        dict: (key,value)
    """
    args.precision = 16 if args.gpu == 0 else 32  # use bfloat16 on RTX 3090
    args.devices = os.cpu_count() if args.accelerator == "cpu" else [args.gpu]
    args.experiment_name = model_experiment_dict[args.model_name]
    args.output_path = str(Path(args.ckpt_path) / "../evaluation")
    args.ckpt_path = get_latest_checkpoint(args.ckpt_path)
    return args


args = parse_evaluation_arguments()
args = update_args(args)
print(args)

test_transform = get_transform_from_model_name(
    args.model_name, image_size=args.image_size, resolution=args.res
)

test_loader = DataLoader(
    get_dataset(args.testpaths, transforms=test_transform),
    batch_size=args.batch_size,
    num_workers=args.num_workers,
    shuffle=False,
    drop_last=False,
)

nifti_saver = NiftiPredictionWriter(output_dir=args.output_path, write_interval="batch")
metrics_saver = MetricsLogger(
    output_dir=args.output_path,
    voxel_spacing=args.res,
    nsd_tolerance=args.nsd_tolerance,
)
evaluation_callbacks = [nifti_saver, metrics_saver]

model_architecture = get_model(model_name=args.model_name, image_size=args.image_size)
model_module: pl.LightningModule = getattr(
    XrayTo3DShape.experiments, args.experiment_name
)(model=model_architecture)
trainer = pl.Trainer(
    callbacks=evaluation_callbacks, accelerator=args.accelerator, devices=args.devices
)
trainer.predict(
    model=model_module,
    dataloaders=test_loader,
    return_predictions=False,
    ckpt_path=args.ckpt_path,
)
