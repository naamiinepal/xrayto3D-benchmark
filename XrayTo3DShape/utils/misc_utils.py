"""utils that do not belong anywhere else or do not require separate module yet"""
from pathlib import Path
import wandb


def get_anatomy_from_path(path: str):
    """used to infer dataset anatomy from filepath
    for example
        configs/paths/lidc/LIDC-IDRI-DRR-full_train+val.csv -> vertebra
    return "none" if unknown path is provided
    """
    anatomies = ["rib", "femur", "hip"]
    vertebra_dataset = ["verse", "lidc", "rsna"]
    # check if the anatomy is mentioned in the path
    for anat in anatomies:
        if anat in path.lower():
            return anat
    # special case for vertebra dataset
    for keyword in vertebra_dataset:
        if keyword in path.lower():
            return "vertebra"
    return "none"


def get_run_from_model_name(model_name, wandb_runs):
    """return the first wandb run with given model name"""
    for run in wandb_runs:
        if model_name in run.config["MODEL_NAME"]:  # allow partial match
            return run
    raise ValueError(f"{model_name} not found")


def filter_wandb_run(
    anatomy: str,
    project_name="msrepo/2d-3d-benchmark",
    tags=("model-compare",),
    verbose=False,
):
    """find wandb runs that fulfil given criteria"""
    api = wandb.Api()
    runs = api.runs(project_name, filters={"tags": {"$in": tags}})
    if verbose:
        print(f"found {len(runs)} unfiltered runs")

    filtered_runs = [
        run
        for run in runs
        if "ANATOMY" in run.config and run.config["ANATOMY"] == anatomy
    ]

    if verbose:
        print(f"got {len(filtered_runs)} runs after filtering anatomy={anatomy}")

    return filtered_runs


def get_latest_checkpoint(path, checkpoint_regex="epoch=*.ckpt"):
    """get latest model checkpoint based on file creation date"""
    checkpoints = list(Path(path).glob(checkpoint_regex))
    latest_checkpoint_path = max(checkpoints, key=lambda x: x.lstat().st_ctime)
    return str(latest_checkpoint_path)


if __name__ == "__main__":
    paths = [
        "configs/paths/totalsegmentator_ribs/TotalSegmentor-ribs-DRR-full_train+val.csv",
        "configs/paths/verse19/Verse2019-DRR-full_train.csv",
        "configs/paths/totalsegmentator_hips/TotalSegmentor-hips-DRR-full_train.csv",
        "configs/paths/rsna_cervical_fracture/RSNACervicalFracture-DRR-full_test.csv",
        "configs/paths/lidc+verse19/LIDC-IDRI-DRR-full_test.csv",
        "configs/paths/femur/30k/TotalSegmentor-femur-left-DRR-30k_test.csv",
        "configs/paths/lidc/LIDC-IDRI-DRR-full_train+val.csv",
    ]
    for path in paths:
        anatomy = get_anatomy_from_path(path)
        print(anatomy, Path(path).stem)
