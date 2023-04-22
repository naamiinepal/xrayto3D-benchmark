import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

import wandb
from XrayTo3DShape import filter_wandb_run, get_run_from_model_name

parser = argparse.ArgumentParser()
parser.add_argument('--anatomy',required=True)
parser.add_argument('--tags',nargs='*')
args = parser.parse_args()

wandb.login()
runs = filter_wandb_run(anatomy=args.anatomy, tags=args.tags,verbose=False)
[print(r.id, r.config['MODEL_NAME']) for r in runs]

if len(runs) == 0:
    print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}. exiting ...")
    sys.exit()

SUBDIR='evaluation'
EVAL_LOG_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/{subdir}/metric-log.csv"
OUT_LOG_CSV_PATH_TEMPLATE = 'results/benchmarking/{anatomy}/{tag}/{model_name}/metric-log.csv'
MODEL_NAMES = [
    "UNETR",
    "AttentionUnet",
    "UNet",
    "MultiScale2DPermuteConcat",
    "TwoDPermuteConcat",
    "OneDConcat",
    "TLPredictor",
]

dataframes = []
for model in MODEL_NAMES:
    try:
        run = get_run_from_model_name(model, runs)
        eval_logfile = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id, subdir=SUBDIR)
        if Path(eval_logfile).exists():
            print(eval_logfile)
            # copy to results directory
            out_logfile = OUT_LOG_CSV_PATH_TEMPLATE.format(anatomy=args.anatomy, tag=args.tags[0],model_name=model)
            print(Path(out_logfile))
            Path(out_logfile).parent.mkdir(exist_ok=True, parents=True)
            shutil.copy(eval_logfile, out_logfile)
            # read csv
            df = pd.read_csv(eval_logfile)
            dataframes.append(df)
            df['model'] = model
    except (ValueError) as e:
        pass

# save into a single csv
MERGED_DF_FILENAME_TEMPLATE = 'results/challengeR/{anatomy}/{tag}/metric-log.csv'
merged_df_out_file = MERGED_DF_FILENAME_TEMPLATE.format(anatomy=args.anatomy, tag=args.tags[0])
Path(merged_df_out_file).parent.mkdir(parents=True, exist_ok=True)
merged_df = pd.concat(dataframes, join='inner').sort_index()
merged_df.to_csv(merged_df_out_file, header=True,index=False)
