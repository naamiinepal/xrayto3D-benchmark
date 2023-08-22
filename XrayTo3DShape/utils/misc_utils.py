"""utils that do not belong anywhere else or do not require separate module yet"""
import re
from pathlib import Path
from typing import Tuple
import wandb


def split_subject_vertebra_id(filepath) -> Tuple[str, str]:
    """
    accomodate these filenames too:
     sub-verse061_22_seg-vert_msk.nii.gz (normal filenames)
     sub-verse401_10_split-verse253_ct.tiff (ignore the first part)
    """
    original_filepath = filepath
    if isinstance(filepath, str):
        filepath = Path(filepath)

    match = re.findall(r"\d+", filepath.name)  # find numbers from string
    ids = list(map(int, match))
    if len(ids) == 2:
        subject_id, vertebra_id = ids
    elif len(ids) == 3:
        _, subject_id, vertebra_id = ids
    else:
        raise ValueError(
            f"could not split {Path(original_filepath).name} into subject and vertebra. got {ids}"
        )
    return subject_id, vertebra_id


def get_anatomy_from_path(path: str):
    """used to infer dataset anatomy from filepath
    for example
        configs/paths/lidc/LIDC-IDRI-DRR-full_train+val.csv -> vertebra
    return "none" if unknown path is provided
    """
    anatomies = ["rib", "femur", "hip"]
    vertebra_dataset = ["verse", "lidc", "rsna", "vertebra"]
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
        if model_name == run.config["MODEL_NAME"]:
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
    filters_mongodb_query_operation ={}
    if len(tags) <= 1:
        filters_mongodb_query_operation["tags"] = {"$in": tags}
    else:
        filters_mongodb_query_operation["$and"] = [{"tags":{"$in":[k]}} for k in tags]
    runs = api.runs(project_name, filters=filters_mongodb_query_operation)
    if verbose:
        print(f"found {len(runs)} unfiltered runs")

    filtered_runs = [
        run
        for run in runs
        if "ANATOMY" in run.config and anatomy in run.config["ANATOMY"]
    ]

    if verbose:
        print(f"got {len(filtered_runs)} runs after filtering anatomy={anatomy}")

    return filtered_runs


def get_latest_checkpoint(path, checkpoint_regex="epoch=*.ckpt"):
    """get latest model checkpoint based on file creation date"""
    checkpoints = list(Path(path).glob(checkpoint_regex))
    latest_checkpoint_path = max(checkpoints, key=lambda x: x.lstat().st_ctime)
    return str(latest_checkpoint_path)


def modify_checkpoint_keys(checkpoint):
    '''amend keys starting with "model" 
    This may be used to load model architecture without using the wrapping object of type BaseExperiment that stores
    the architecture as model variable'''
    for key in list(checkpoint["state_dict"].keys()):
        # model.layer1.conv1 -> layer1.conv1
        if str(key).startswith("model."):
            modified_key = str(key)[len("model.") :]
            value = checkpoint["state_dict"].pop(key)
            checkpoint["state_dict"][modified_key] = value
    return checkpoint 

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
