# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""
run inference using model checkpoint.
save metrics to csv log, predictions as nifti
"""
import argparse
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

import XrayTo3DShape
from XrayTo3DShape import (
    MetricsLogger,
    AnglePerturbationMetricsLogger,
    NiftiPredictionWriter,
    get_dataset,
    get_latest_checkpoint,
    get_model,
    get_transform_from_model_name,
    model_experiment_dict,
    TLPredictorExperiment,
    CustomAutoEncoder,
    anatomy_resolution_dict,
    get_anatomy_from_path,
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
    parser.add_argument("--ckpt_type", choices=["latest", "best"], default="latest")
    parser.add_argument("--res", type=float)
    parser.add_argument("--load_autoencoder_from", type=str)
    parser.add_argument("--nsd_tolerance", type=float, default=1.5)
    parser.add_argument("--image_size", type=int)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--precision", default=32)
    parser.add_argument("--angle_perturbation", default=False, action="store_true")
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
    if args.output_path is None:
        args.output_path = str(Path(args.ckpt_path) / "../evaluation")
    if args.ckpt_type == "best":
        args.ckpt_path = get_latest_checkpoint(
            args.ckpt_path, checkpoint_regex="epoch=*.ckpt"
        )
    elif args.ckpt_type == "latest":
        args.ckpt_path = get_latest_checkpoint(
            args.ckpt_path, checkpoint_regex="last*.ckpt"
        )
    else:
        raise ValueError(
            f"ckpt_type can be either `best` or `latest` but got {args.ckpt_type}"
        )
    # assert resolution and size agree for each anatomy
    args.anatomy = get_anatomy_from_path(args.testpaths)
    # this requirement does not make sense when data is a patch 
    # orig_size, orig_res = anatomy_resolution_dict[args.anatomy]
    # assert int(args.image_size * args.res) == int(
    #     orig_size * orig_res
    # ), f"({args.image_size},{args.res}) does not match ({orig_size},{orig_res})"
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

nifti_saver = NiftiPredictionWriter(
    output_dir=args.output_path,
    write_interval="batch",
    image_size=args.image_size,
    resolution=args.res,
)
if args.angle_perturbation:
    metrics_saver = AnglePerturbationMetricsLogger(
        output_dir=args.output_path,
        voxel_spacing=args.res,
        nsd_tolerance=args.nsd_tolerance,
    )
else:
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

if args.experiment_name == TLPredictorExperiment.__name__:
    print(f"loading autoencoder from {args.load_autoencoder_from}")
    ae_model = get_model(
        model_name=CustomAutoEncoder.__name__, image_size=args.image_size
    )
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

    ae_model.load_state_dict(checkpoint["state_dict"], strict=True)
    model_module.set_decoder(ae_model)  # type: ignore

    # load model architecture
    if Path(args.ckpt_path).exists():
        checkpoint = torch.load(args.ckpt_path)
    else:
        raise ValueError(f"model checkpoint {args.ckpt_path} does not exist")
    for key in list(checkpoint["state_dict"].keys()):
        # model.layer1.conv1 -> layer1.conv1
        if str(key).startswith("model."):
            modified_key = str(key)[len("model.") :]
            value = checkpoint["state_dict"].pop(key)
            checkpoint["state_dict"][modified_key] = value
    if "loss_function.pos_weight" in checkpoint["state_dict"]:
        checkpoint["state_dict"].pop("loss_function.pos_weight")
    print(checkpoint["state_dict"].keys())

    model_architecture.load_state_dict(checkpoint["state_dict"], strict=False)
    model_module.model = model_architecture

trainer = pl.Trainer(
    callbacks=evaluation_callbacks, accelerator=args.accelerator, devices=args.devices
)
trainer.predict(
    model=model_module,
    dataloaders=test_loader,
    return_predictions=False,
    ckpt_path=None
    if args.experiment_name == TLPredictorExperiment.__name__
    else args.ckpt_path,
)
