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
    parser.add_argument("--domain_shift", default=False, action="store_true")
    parser.add_argument("--domain_shift_dataset")
    parser.add_argument("--tags", nargs="*")
    parser.add_argument("--save_json", default=False, action="store_true")
    args = parser.parse_args()

    print(args)
    if args.domain_shift:
        subdir = f"domain_shift_{args.domain_shift_dataset}"
    else:
        subdir = "evaluation"
    EVAL_LOG_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/{subdir}/metric-log.csv"

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

    # print latex table
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
    latex_table_row_template = r" & {model_name} & {model_size} & {DSC:.2f}  & {HD95:.2f} & {ASD:.2f}  & {NSD:.2f} \\"  # make this a raw string so that two backslashes \\ are not escaped and printed as is

    latex_table = ""
    model_dsc_dict = {}
    for model in MODEL_NAMES:
        try:
            run = get_run_from_model_name(model, runs)
            # read metric log csv
            csv_filename = EVAL_LOG_CSV_PATH_TEMPLATE.format(
                run_id=run.id, subdir=subdir
            )
            print(f"reading {csv_filename}")
            df = pd.read_csv(csv_filename)
            df.replace([np.inf,-np.inf],np.nan,inplace=True) # replace inf with nan so that they can be dropped when aggregating later
            latex_table += latex_table_row_template.format(
                model_name=run.config["MODEL_NAME"],
                DSC=df.mean(numeric_only=True,skipna=True).DSC * 100,
                HD95=df.mean(numeric_only=True,skipna=True).HD95,
                ASD=df.mean(numeric_only=True,skipna=True).ASD,
                NSD=df.mean(numeric_only=True,skipna=True).NSD,
                model_size=model_sizes[model],
            )
            model_dsc_dict[run.config["MODEL_NAME"]] = df.mean(numeric_only=True).DSC
        except (ValueError, FileNotFoundError) as e:
            latex_table += latex_table_row_template.format(
                model_name=model,
                DSC=float("nan"),
                HD95=float("nan"),
                ASD=float("nan"),
                NSD=float("nan"),
                model_size=model_sizes[model],
            )

    print(latex_table)
    if args.save_json:
        json_outpath = (
            f"domainshift_results/{args.anatomy}_outdomain_{args.domain_shift_dataset}.json"
            if args.domain_shift
            else f"domainshift_results/{args.anatomy}_indomain.json"
        )
        with open(json_outpath, "w") as fp:
            json.dump(model_dsc_dict, fp)
