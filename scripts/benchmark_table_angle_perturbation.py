import json
import sys
import numpy as np

import pandas as pd

import wandb
from XrayTo3DShape import filter_wandb_run, get_run_from_model_name

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy", required=True)
    parser.add_argument("--tags", nargs="*")
    parser.add_argument("--save_json", default=False, action="store_true")
    args = parser.parse_args()

    print(args)
    subdir = "angle_perturbation"
    EVAL_LOG_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/angle_perturbation/{angle_perturbation}/metric-log.csv"
    EVAL_LOG_NORMAL_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/evaluation/metric-log.csv"
    # extract wandb runs
    wandb.login()

    runs = filter_wandb_run(anatomy=args.anatomy, tags=args.tags, verbose=True)
    for r in runs:
        print(r.id, r.config["MODEL_NAME"])

    if len(runs) == 0:
        print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}. exiting ...")
        sys.exit()
    else:
        print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}.")

    MODEL_NAMES = [
        "SwinUNETR",
        "AttentionUnet",
        "TwoDPermuteConcat",
        "UNet",
        "MultiScale2DPermuteConcat",
        "UNETR",
        "TLPredictor",
        "OneDConcat",
    ]
    model_sizes = {
        "SwinUNETR": "62.2M",
        "AttentionUnet": "1.5M",
        "UNet": "1.2M",
        "MultiScale2DPermuteConcat": "3.5M",
        "TwoDPermuteConcat": "1.2M",
        "OneDConcat": "40.6M",
        "TLPredictor": "6.6M",
        "UNETR": "96.2M",
    }

    model_dsc_dict = {}
    for model in MODEL_NAMES:
        try:
            run = get_run_from_model_name(model, runs)
            model_name=run.config["MODEL_NAME"]
            # read non-perturbed metric log csv
            normal_csv_filename =EVAL_LOG_NORMAL_CSV_PATH_TEMPLATE.format(run_id=run.id)
            df = pd.read_csv(normal_csv_filename)
            df.replace([np.inf,-np.inf],np.nan,inplace=True) # replace inf with nan so that they can be dropped when aggregating later
            DSC=df.mean(numeric_only=True,skipna=True).DSC * 100
            HD95=df.mean(numeric_only=True,skipna=True).HD95
            ASD=df.mean(numeric_only=True,skipna=True).ASD
            NSD=df.mean(numeric_only=True,skipna=True).NSD
            model_dsc_dict[run.config['MODEL_NAME']] = {}
            model_dsc_dict[run.config['MODEL_NAME']]['DSC'] = {}
            model_dsc_dict[run.config['MODEL_NAME']]['HD95'] = {}
            model_dsc_dict[run.config['MODEL_NAME']]['ASD'] = {}
            model_dsc_dict[run.config['MODEL_NAME']]['NSD'] = {}

            model_dsc_dict[run.config["MODEL_NAME"]]['DSC'][str(0)] = DSC
            model_dsc_dict[run.config["MODEL_NAME"]]['HD95'][str(0)] = HD95
            model_dsc_dict[run.config["MODEL_NAME"]]['ASD'][str(0)] = ASD
            model_dsc_dict[run.config["MODEL_NAME"]]['NSD'][str(0)] = NSD       

            ANGLE_PERTURBATIONS = [1,2,5,10]
            for angle in ANGLE_PERTURBATIONS:
                csv_filename = EVAL_LOG_CSV_PATH_TEMPLATE.format(
                    run_id=run.id, subdir=subdir, angle_perturbation=angle
                )
                print(f"reading {csv_filename}")
                df = pd.read_csv(csv_filename)
                df.replace([np.inf,-np.inf],np.nan,inplace=True) # replace inf with nan so that they can be dropped when aggregating later
                DSC=df.mean(numeric_only=True,skipna=True).DSC * 100,
                HD95=df.mean(numeric_only=True,skipna=True).HD95,
                ASD=df.mean(numeric_only=True,skipna=True).ASD,
                NSD=df.mean(numeric_only=True,skipna=True).NSD,
                model_size=model_sizes[model]
                model_dsc_dict[run.config["MODEL_NAME"]]['DSC'][str(angle)] = DSC
                model_dsc_dict[run.config["MODEL_NAME"]]['HD95'][str(angle)] = HD95
                model_dsc_dict[run.config["MODEL_NAME"]]['ASD'][str(angle)] = ASD
                model_dsc_dict[run.config["MODEL_NAME"]]['NSD'][str(angle)] = NSD

        except (ValueError, FileNotFoundError, AttributeError) as e:
            print(e)
            
    print(model_dsc_dict)

    if args.save_json:
        json_outpath = (
            f"angle_perturbation_results/{args.anatomy}_angle_perturbation.json"
        )
        with open(json_outpath, "w") as fp:
            json.dump(model_dsc_dict, fp)
