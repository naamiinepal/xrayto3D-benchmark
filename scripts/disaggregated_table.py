"""additional metadata"""
import sys
from pathlib import Path

import pandas as pd

import wandb
from XrayTo3DShape import (
    MODEL_NAMES,
    VerseExcelSheet,
    VerseKeys,
    filter_wandb_run,
    get_run_from_model_name,
)


def get_shape(row):
    """shape lambda"""
    return verse_metadata.get_shape(verse_metadata.get_vertebra_keys(row["subject-id"]))


def get_severity(row):
    """severity lambda"""
    return verse_metadata.get_severity(
        verse_metadata.get_vertebra_keys(row["subject-id"])
    )


def get_vertebra_level(row):
    """vertebra level"""
    return verse_metadata.get_vertebra_level(
        verse_metadata.get_vertebra_keys(row["subject-id"])
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--anatomy", required=True)
    parser.add_argument("--tags", nargs="*")
    args = parser.parse_args()

    EVAL_LOG_CSV_PATH_TEMPLATE = "/mnt/SSD0/mahesh-home/xrayto3D-benchmark/runs/2d-3d-benchmark/{run_id}/evaluation/metric-log.csv"

    verse_metadata = VerseExcelSheet()
    # extract wandb runs
    wandb.login()
    runs = filter_wandb_run(anatomy=args.anatomy, tags=args.tags, verbose=False)

    print(f"found {len(runs)} wandb runs")
    if len(runs) == 0:
        print(f"found {len(runs)} wandb runs for anatomy {args.anatomy}. exiting ...")
        sys.exit()
    # write additional metadata to table
    for model in MODEL_NAMES:
        try:
            run = get_run_from_model_name(model, runs)
            # read metric log csv
            input_log = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id)
            df = pd.read_csv(input_log)
            df["shape"] = df.apply(get_shape, axis="columns")
            df["severity"] = df.apply(get_severity, axis="columns")
            df["level"] = df.apply(get_vertebra_level, axis="columns")
            out_log = Path(input_log).with_name("metric_log_metadata.csv")
            df.to_csv(out_log, index=False)

        except ValueError as e:
            pass

    # print vertebra level disaggregation
    for model in MODEL_NAMES:
        try:
            run = get_run_from_model_name(model, runs)
            input_log = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id)
            out_log = Path(input_log).with_name("metric_log_metadata.csv")
            df = pd.read_csv(out_log)
            row = f"{model:20s}"
            for metric_type in ["DSC", "HD95", "NSD"]:
                for level_type in [
                    VerseKeys.CERVICAL,
                    VerseKeys.THORACIC,
                    VerseKeys.LUMBAR,
                ]:
                    row += f'& {df.groupby(["level"])[metric_type].mean()[level_type]:5.2f}'
            row += r"\\"  # latex new line
            print(row)

        except ValueError as e:
            pass
    print("\n")

    # print vertebra shape disaggregation
    for model in MODEL_NAMES:
        try:
            run = get_run_from_model_name(model, runs)
            input_log = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id)
            out_log = Path(input_log).with_name("metric_log_metadata.csv")
            df = pd.read_csv(out_log)
            row = f"{model:20s}"
            for metric_type in ["DSC", "HD95", "NSD"]:
                for level_type in [
                    VerseKeys.NORMAL,
                    VerseKeys.WEDGE,
                    VerseKeys.BICONCAVE,
                    VerseKeys.CRUSH,
                ]:
                    row += f'& {df.groupby(["shape"])[metric_type].mean()[level_type]:5.2f}'
            row += r"\\"  # latex new line
            print(row)

        except ValueError as e:
            pass
    print("\n")

    # print vertebra severity disaggregation
    for model in MODEL_NAMES:
        try:
            run = get_run_from_model_name(model, runs)
            input_log = EVAL_LOG_CSV_PATH_TEMPLATE.format(run_id=run.id)
            out_log = Path(input_log).with_name("metric_log_metadata.csv")
            df = pd.read_csv(out_log)
            row = f"{model:20s}"
            for metric_type in ["DSC", "HD95", "NSD"]:
                for level_type in [
                    VerseKeys.NORMAL,
                    VerseKeys.MILD,
                    VerseKeys.MODERATE,
                    VerseKeys.SEVERE,
                ]:
                    row += f'& {df.groupby(["severity"])[metric_type].mean()[level_type]:5.2f}'
            row += r"\\"  # latex new line
            print(row)
        except ValueError as e:
            pass
    print("\n")
