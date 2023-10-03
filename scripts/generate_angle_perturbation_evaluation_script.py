"""generate evaluation script"""
from pathlib import Path
import sys
import argparse
import wandb
from XrayTo3DShape import (
    filter_wandb_run,
    get_run_from_model_name,
    get_anatomy_from_path,
)

expt_dict = {
    "OneDConcat": "ParallelHeadsExperiment",
    "MultiScale2DPermuteConcat": "ParallelHeadsExperiment",
    "TwoDPermuteConcat": "ParallelHeadsExperiment",
    "AttentionUnet": "VolumeAsInputExperiment",
    "UNet": "VolumeAsInputExperiment",
    "UNETR": "VolumeAsInputExperiment",
    "SwinUNETR": "VolumeAsInputExperiment",
}


parser = argparse.ArgumentParser()
parser.add_argument("--testpaths")
parser.add_argument("-ckpt_type", choices=["best", "latest"], default="latest")
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--img_size")
parser.add_argument("--res")
parser.add_argument("--tags", nargs="*")
parser.add_argument('--debug',default=False,action='store_true')


args = parser.parse_args()

anatomy = get_anatomy_from_path(args.testpaths)

# extract wandb runs
wandb.login()
runs = filter_wandb_run(anatomy=anatomy, tags=args.tags, verbose=False)

if args.debug:
    print(f'found {len(runs)} wandb runs for anatomy {get_anatomy_from_path(args.testpaths)}')
    for run in runs:
        print(run.id,run.config["MODEL_NAME"])
        
if len(runs) == 0:
    print(f"found {len(runs)} wandb runs for anatomy {get_anatomy_from_path(args.testpaths)}. exiting ...")
    sys.exit()
CKPT_PATH_TEMPLATE = "runs/2d-3d-benchmark/{run_id}/checkpoints"
for model_name in expt_dict:
    run = get_run_from_model_name(model_name, runs)
    ckpt_path = CKPT_PATH_TEMPLATE.format(run_id=run.id)

    BATCH_SIZE  = 1 if model_name in ['UNETR','SwinUNETR'] else args.batch_size
    ANGLE_PERTURBATIONS = [1,2,5,10]
    for angle in ANGLE_PERTURBATIONS:
        metric_log_output_path = f"{ckpt_path}/../angle_perturbation/{angle}"
        test_csv_path = args.testpaths
        angle_perturbation_csv_path = Path(test_csv_path).with_name(Path(test_csv_path).stem + '_' + str(angle)).with_suffix(".csv")

        
        command = f"python evaluate.py  --testpaths {angle_perturbation_csv_path} --gpu {args.gpu} --image_size {args.img_size} --batch_size {BATCH_SIZE} --accelerator gpu --res {args.res} --model_name {model_name} --ckpt_path {ckpt_path} --ckpt_type {args.ckpt_type} --gpu {args.gpu} --output_path {metric_log_output_path}\n"
        if not args.debug:
            print(command)
