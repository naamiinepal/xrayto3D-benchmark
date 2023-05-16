"""generate evaluation script"""
import sys
import wandb
from XrayTo3DShape import filter_wandb_run, MODEL_NAMES, get_run_from_model_name


ANATOMY = "vertebra"
tags = ["dropout", "model-compare"]
wandb.login()
runs = filter_wandb_run(anatomy=ANATOMY, tags=tags, verbose=False)
if len(runs) == 0:
    print(f"found {len(runs)} wandb runs for anatomy {ANATOMY}. exiting ...")
    sys.exit()
for model_name in MODEL_NAMES:
    run = get_run_from_model_name(model_name, runs)

    command_csv_generate = f"python external/xrayto3D-morphometry/vertebra_landmarks.py --dir runs/2d-3d-benchmark/{run.id}/evaluation --log_filename vertebra_morphometry.csv"
    command_csv_processing = f"python external/xrayto3D-morphometry/scripts/process_vertebra_csv.py runs/2d-3d-benchmark/{run.id}/evaluation/vertebra_morphometry.csv"

    print(f"# model type: {model_name} tags {tags}")
    print(command_csv_generate)
    print(command_csv_processing)
