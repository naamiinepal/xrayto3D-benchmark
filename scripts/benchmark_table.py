import sys

import pandas as pd

import wandb
from XrayTo3DShape import filter_wandb_run, get_run_from_model_name

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy", required=True)
    parser.add_argument("--tags", nargs="*")
    args = parser.parse_args()

    EVAL_LOG_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/evaluation/metric-log.csv"

    # extract wandb runs
    wandb.login()
    runs = filter_wandb_run(anatomy=args.anatomy, tags=args.tags, verbose=False)
    for r in runs:
        print(r.id, r.config["MODEL_NAME"])

    if len(runs) == 0:
        print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}. exiting ...")
        sys.exit()

    # print latex table
    MODEL_NAMES = [
        "UNETR",
        "AttentionUnet",
        "UNet",
        "MultiScale2DPermuteConcat",
        "TwoDPermuteConcat",
        "OneDConcat",
        "TLPredictor",
    ]
    model_sizes = {
        "AttentionUnet": "1.5M",
        "UNet": "1.2M",
        "MultiScale2DPermuteConcat": "3.5M",
        "TwoDPermuteConcat": "1.2M",
        "OneDConcat": "40.6M",
        "TLPredictor": "6.6M",
    }
    latex_table_row_template = r" & {model_name} & {model_size} & {DSC:.2f}  & {HD95:.2f} & {ASD:.2f}  & {NSD:.2f} \\"  # make this a raw string so that two backslashes \\ are not escaped and printed as is

    latex_table = ""
    for model in MODEL_NAMES:
        try:
            run = get_run_from_model_name(model, runs)
            # read metric log csv
            df = pd.read_csv(EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id))
            latex_table += latex_table_row_template.format(
                model_name=run.config["MODEL_NAME"],
                DSC=df.mean(numeric_only=True).DSC,
                HD95=df.mean(numeric_only=True).HD95,
                ASD=df.mean(numeric_only=True).ASD,
                NSD=df.mean(numeric_only=True).NSD,
                model_size=model_sizes[model],
            )
        except ValueError as e:
            latex_table += latex_table_row_template.format(
                model_name=model,
                DSC=float("nan"),
                HD95=float("nan"),
                ASD=float("nan"),
                NSD=float("nan"),
                model_size=model_sizes[model],
            )

    print(latex_table)
