import sys
import argparse
import wandb
from XrayTo3DShape import (
    filter_wandb_run,
    get_run_from_model_name,
    get_anatomy_from_path,
    get_latest_checkpoint,
)

expt_dict = {
    "OneDConcat": "ParallelHeadsExperiment",
    "MultiScale2DPermuteConcat": "ParallelHeadsExperiment",
    "TwoDPermuteConcat": "ParallelHeadsExperiment",
    "AttentionUnet": "VolumeAsInputExperiment",
    "UNet": "VolumeAsInputExperiment",
}


parser = argparse.ArgumentParser()
parser.add_argument("--testpaths")
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--img_size")
parser.add_argument("--res")


args = parser.parse_args()

anatomy = get_anatomy_from_path(args.testpaths)
# extract wandb runs
wandb.login()
runs = filter_wandb_run(anatomy=anatomy, verbose=False)

if len(runs) == 0:
    print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}. exiting ...")
    sys.exit()
ckpt_path_template = "runs/2d-3d-benchmark/{run_id}/checkpoints"
for model_name in expt_dict:
    run = get_run_from_model_name(model_name, runs)
    ckpt_path = ckpt_path_template.format(run_id=run.id)
    command = f"python evaluate.py  --testpaths {args.testpaths} --gpu {args.gpu} --image_size {args.img_size} --batch_size {args.batch_size} --accelerator gpu --res {args.res} --model_name {model_name} --ckpt_path {ckpt_path} --gpu {args.gpu}\n"
    print(command)
