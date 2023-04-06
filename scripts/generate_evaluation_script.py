"""generate evaluation script"""
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
}


parser = argparse.ArgumentParser()
parser.add_argument("--testpaths")
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--img_size")
parser.add_argument("--res")
parser.add_argument("--tags", nargs="*")
parser.add_argument("--domain_shift", default=False, action="store_true")
parser.add_argument("--domain_shift_dataset", default="", required=False)


args = parser.parse_args()


anatomy = get_anatomy_from_path(args.testpaths)

# extract wandb runs
wandb.login()
runs = filter_wandb_run(anatomy=anatomy, tags=args.tags, verbose=False)

if len(runs) == 0:
    print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}. exiting ...")
    sys.exit()
CKPT_PATH_TEMPLATE = "runs/2d-3d-benchmark/{run_id}/checkpoints"
for model_name in expt_dict:
    run = get_run_from_model_name(model_name, runs)
    ckpt_path = CKPT_PATH_TEMPLATE.format(run_id=run.id)
    metric_log_output_path = (
        f"{ckpt_path}/../domain_shift_{args.domain_shift_dataset}"
        if args.domain_shift
        else f"{ckpt_path}/../evaluation"
    )

    command = f"python evaluate.py  --testpaths {args.testpaths} --gpu {args.gpu} --image_size {args.img_size} --batch_size {args.batch_size} --accelerator gpu --res {args.res} --model_name {model_name} --ckpt_path {ckpt_path} --gpu {args.gpu} --output_path {metric_log_output_path}\n"
    print(command)
