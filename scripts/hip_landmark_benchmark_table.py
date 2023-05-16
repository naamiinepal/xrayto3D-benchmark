import pandas as pd
import wandb
from XrayTo3DShape import filter_wandb_run, get_run_from_model_name

MODEL_NAMES = [
    "SwinUNETR",
    "UNETR",
    "AttentionUnet",
    "UNet",
    "MultiScale2DPermuteConcat",
    "TwoDPermuteConcat",
    "OneDConcat",
    "TLPredictor",
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

ANATOMY = "hip"
tags = ["dropout", "model-compare"]
wandb.login()
runs = filter_wandb_run(anatomy=ANATOMY, tags=tags, verbose=False)
latex_table_row_template = r" & {model_name} & {model_size} & {ASIS_L:.2f}  & {ASIS_R:.2f}  & {PT_L:.2f} & {PT_R:.2f}  & {IS_L:.2f}  & {IS_R:.2f}  & {PSIS_L:.2f}  & {PSIS_R:.2f}\\"  # make this a raw string so that two backslashes \\ are not escaped and printed as is
EVAL_LOG_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/{subdir}/hip_landmark_error.csv"
subdir = "evaluation"

latex_table = ""

for model_name in MODEL_NAMES:
    run = get_run_from_model_name(model_name, runs)
    csv_filename = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id, subdir=subdir)
    df = pd.read_csv(csv_filename)
    print(df)
    latex_table += latex_table_row_template.format(
        model_name=run.config["MODEL_NAME"],
        ASIS_L=df.median(numeric_only=True).ASIS_L,
        ASIS_R=df.median(numeric_only=True).ASIS_R,
        PT_L=df.median(numeric_only=True).PT_L,
        PT_R=df.median(numeric_only=True).PT_R,
        IS_L=df.median(numeric_only=True).IS_L,
        IS_R=df.median(numeric_only=True).IS_R,
        PSIS_L=df.median(numeric_only=True).PSIS_L,
        PSIS_R=df.median(numeric_only=True).PSIS_R,
        model_size=model_sizes[model_name],
    )
print(latex_table)
