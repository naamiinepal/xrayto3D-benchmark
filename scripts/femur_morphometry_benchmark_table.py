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

ANATOMY = 'femur'
tags = ['dropout', 'model-compare']
wandb.login()
runs = filter_wandb_run(anatomy=ANATOMY, tags=tags, verbose=False)
latex_table_row_template = r" & {model_name} & {model_size} & {FHR:.2f}  & {FHC:.2f}  & {NSA:.2f} & {FNA_x:.2f}  & {FNA_y:.2f}  & {FNA_z:.2f}  & {FDA_x:.2f}  & {FDA_y:.2f} & {FDA_z:.2f}\\"  # make this a raw string so that two backslashes \\ are not escaped and printed as is
EVAL_LOG_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/{subdir}/femur_morphometry_error.csv"
subdir = "evaluation"

latex_table = ""

for model_name in MODEL_NAMES:
    run = get_run_from_model_name(model_name, runs)
    csv_filename = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id,subdir=subdir)
    df = pd.read_csv(csv_filename)
    print(df)
    latex_table += latex_table_row_template.format(
        model_name=run.config["MODEL_NAME"],
        FHR=df.mean(numeric_only=True).FHR,
        FHC=df.mean(numeric_only=True).FHC,
        NSA=df.mean(numeric_only=True).NSA,
        FNA_x=df.mean(numeric_only=True).FNA_x,
        FNA_y=df.mean(numeric_only=True).FNA_y,
        FNA_z=df.mean(numeric_only=True).FNA_z,
        FDA_x=df.mean(numeric_only=True).FDA_x,
        FDA_y=df.mean(numeric_only=True).FDA_y,
        FDA_z=df.mean(numeric_only=True).FDA_z,
        model_size=model_sizes[model_name],
    )
print(latex_table)    