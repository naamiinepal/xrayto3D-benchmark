"""generate evaluation script"""
import sys
import argparse
import wandb
from XrayTo3DShape import (
    filter_wandb_run,
    get_run_from_model_name,
)

expt_dict = {
    "OneDConcat": "ParallelHeadsExperiment",
    "MultiScale2DPermuteConcat": "ParallelHeadsExperiment",
    "TwoDPermuteConcat": "ParallelHeadsExperiment",
    "AttentionUnet": "VolumeAsInputExperiment",
    "UNet": "VolumeAsInputExperiment",
    "UNETR": "VolumeAsInputExperiment",
}


parser = argparse.ArgumentParser()
parser.add_argument("--tags", nargs="*")
parser.add_argument("--iterative", default=False, action="store_true")

args = parser.parse_args()

ANATOMY = "hip"
# extract wandb runs
wandb.login()
runs = filter_wandb_run(anatomy=ANATOMY, tags=args.tags, verbose=False)

if len(runs) == 0:
    print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}. exiting ...")
    sys.exit()
NIFTI_DIR_TEMPLATE = "runs/2d-3d-benchmark/{run_id}/evaluation"
for model_name in expt_dict:
    run = get_run_from_model_name(model_name, runs)
    nifti_dir = NIFTI_DIR_TEMPLATE.format(run_id=run.id)
    iterative_switch = "--iterative" if args.iterative else ""
    command = f"python external/xrayto3D-morphometry/pelvic_landmarks_main.py --dir {nifti_dir} --log_filename morphometry_gt.csv {iterative_switch}\n"
    command += f"python external/xrayto3D-morphometry/pelvic_landmarks_main.py --dir {nifti_dir} --log_filename morphometry_pred.csv --predicted {iterative_switch}\n"
    print(command)
