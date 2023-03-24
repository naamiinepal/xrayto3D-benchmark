"""utils that do not belong anywhere else or do not require separate module yet"""
from collections import ChainMap
import inspect
from pathlib import Path
from monai.data.meta_tensor import MetaTensor
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


def get_latest_checkpoint(path, checkpoint_regex='epoch=*.ckpt'):
    """get latest model checkpoint based on file creation date"""
    checkpoints = list(Path(path).glob(checkpoint_regex))
    latest_checkpoint_path = max(checkpoints, key=lambda x: x.lstat().st_ctime)
    return str(latest_checkpoint_path)


def merge_dicts(dict1, dict2):
    return ChainMap(dict1, dict2)


def printarr(*arrs, float_width=6):
    """
    Print a pretty table giving name, shape, dtype, type, and content information for input tensors or scalars.

    Call like: printarr(my_arr, some_other_arr, maybe_a_scalar). Accepts a variable number of arguments.

    Inputs can be:
        - Numpy tensor arrays
        - Pytorch tensor arrays
        - Jax tensor arrays
        - Python ints / floats
        - additionally monai.data.meta_tensor.MetaTensor
        - None

    It may also work with other array-like types, but they have not been tested.

    Use the `float_width` option specify the precision to which floating point types are printed.

    Author: Nicholas Sharp (nmwsharp.com)
    Canonical source: https://gist.github.com/nmwsharp/54d04af87872a4988809f128e1a1d233
    License: This snippet may be used under an MIT license, and it is also released into the public domain.
             Please retain this docstring as a reference.
    """

    frame = inspect.currentframe().f_back
    default_name = "[temporary]"

    ## helpers to gather data about each array
    def name_from_outer_scope(a):
        if a is None:
            return "[None]"
        name = default_name
        for k, v in frame.f_locals.items():
            if v is a:
                name = k
                break
        return name

    def dtype_str(a):
        if a is None:
            return "None"
        if isinstance(a, int):
            return "int"
        if isinstance(a, float):
            return "float"
        return str(a.dtype)

    def shape_str(a):
        if a is None:
            return "N/A"
        if isinstance(a, int):
            return "scalar"
        if isinstance(a, float):
            return "scalar"
        return str(list(a.shape))

    def type_str(a):
        return str(type(a))[8:-2]  # TODO this is is weird... what's the better way?

    def device_str(a):
        if hasattr(a, "device"):
            device_str = str(a.device)
            if len(device_str) < 10:
                # heuristic: jax returns some goofy long string we don't want, ignore it
                return device_str
        return ""

    def format_float(x):
        return f"{x:{float_width}g}"

    def minmaxmean_str(a):
        if a is None:
            return ("N/A", "N/A", "N/A")
        if isinstance(a, int) or isinstance(a, float):
            return (format_float(a), format_float(a), format_float(a))
        if isinstance(a, MetaTensor):
            a = a.as_tensor()
        # compute min/max/mean. if anything goes wrong, just print 'N/A'
        min_str = "N/A"
        try:
            min_str = format_float(a.min())
        except:
            pass
        max_str = "N/A"
        try:
            max_str = format_float(a.max())
        except:
            pass
        mean_str = "N/A"
        try:
            mean_str = format_float(a.mean())
        except:
            pass

        return (min_str, max_str, mean_str)

    try:
        props = ["name", "dtype", "shape", "type", "device", "min", "max", "mean"]

        # precompute all of the properties for each input
        str_props = []
        for a in arrs:
            minmaxmean = minmaxmean_str(a)
            str_props.append(
                {
                    "name": name_from_outer_scope(a),
                    "dtype": dtype_str(a),
                    "shape": shape_str(a),
                    "type": type_str(a),
                    "device": device_str(a),
                    "min": minmaxmean[0],
                    "max": minmaxmean[1],
                    "mean": minmaxmean[2],
                }
            )

        # for each property, compute its length
        maxlen = {}
        for p in props:
            maxlen[p] = 0
        for sp in str_props:
            for p in props:
                maxlen[p] = max(maxlen[p], len(sp[p]))

        # if any property got all empty strings, don't bother printing it, remove if from the list
        props = [p for p in props if maxlen[p] > 0]

        # print a header
        header_str = ""
        for p in props:
            prefix = "" if p == "name" else " | "
            fmt_key = ">" if p == "name" else "<"
            header_str += f"{prefix}{p:{fmt_key}{maxlen[p]}}"
        print(header_str)
        print("-" * len(header_str))

        # now print the acual arrays
        for strp in str_props:
            for p in props:
                prefix = "" if p == "name" else " | "
                fmt_key = ">" if p == "name" else "<"
                print(f"{prefix}{strp[p]:{fmt_key}{maxlen[p]}}", end="")
            print("")

    finally:
        del frame


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
