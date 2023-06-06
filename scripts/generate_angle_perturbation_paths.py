"""generate data sample filepaths split into train, val, test sets"""
from pathlib import Path
from typing import Sequence, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xrayto3d_preprocess import read_config_and_load_components, read_subject_list


def get_individual_fullpaths(
    subject_dir,
    config,
    perturbation_angle: Union[int, None],
    ap_pattern="*ap.png",
    lat_pattern="*lat.png",
    seg_pattern="*msk.nii.gz",
):
    """return ap, lat, seg filepaths"""
    derivatives_path = (
        Path(config.subjects.subject_basepath).resolve()
        / f"{subject_dir}"
        / "derivatives"
    )
    if perturbation_angle is None:
        xray_lat_basepath = derivatives_path / "xray_from_ct"
    else:
        xray_lat_basepath = (
            derivatives_path
            / "xray_from_ct_angle_perturbation"
            / str(perturbation_angle)
        )
    xray_ap_basepath = derivatives_path / "xray_from_ct"
    seg_roi_basepath = derivatives_path / "seg_roi"
    print(xray_ap_basepath, xray_lat_basepath, seg_roi_basepath)
    xray_ap = sorted(xray_ap_basepath.rglob(ap_pattern))
    xray_lat = sorted(xray_lat_basepath.rglob(lat_pattern))
    seg = sorted(seg_roi_basepath.rglob(seg_pattern))
    print(xray_ap, xray_lat, seg)
    return xray_ap, xray_lat, seg


def get_fullpaths(
    subject_list: Sequence,
    config,
    perturbation_angle,
    ap_pattern,
    lat_pattern,
    seg_pattern,
):
    """return ap, lat, seg paths as dict"""
    print(ap_pattern, lat_pattern, seg_pattern)
    ap, lat, seg = [], [], []
    if config["dataset"] == "verse":
        for subject, subject_dir in subject_list:
            _ap, _lat, _seg = get_individual_fullpaths(
                subject_dir,
                config,
                perturbation_angle,
                ap_pattern,
                lat_pattern,
                seg_pattern,
            )
            ap.extend(_ap)
            lat.extend(_lat)
            seg.extend(_seg)
    else:
        for subject in subject_list:
            _ap, _lat, _seg = get_individual_fullpaths(
                subject,
                config,
                perturbation_angle,
                ap_pattern,
                lat_pattern,
                seg_pattern,
            )
            ap.extend(_ap)
            lat.extend(_lat)
            seg.extend(_seg)
    assert len(ap) == len(
        seg
    ), f"Number of segmentations {len(seg)} and Number of AP xrays {len(ap)} do not match"

    return {"ap": ap, "lat": lat, "seg": seg}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    parser.add_argument("--ap", type=str, default="*hip-ap.png")
    parser.add_argument("--lat", type=str, default="*hip-lat.png")
    parser.add_argument("--seg", type=str, default="*hip_msk.nii.gz")
    # parser.add_argument("--ap", type=str, default="*vert*_ap.png")
    # parser.add_argument("--lat", type=str, default="*vert*_lat.png")
    # parser.add_argument("--seg", type=str, default="*vert_msk.nii.gz")
    args = parser.parse_args()
    print(args)
    dataset = (
        "verse"
        if str(args.config_file).split("-")[0].lower().startswith("verse")
        else "others"
    )

    # config_path = "configs/test/Verse2020-DRR-test.yaml"
    config_path = args.config_file
    # use config to find the fullpath of x-ray and seg_roi pairs

    config = read_config_and_load_components(config_path)
    # the perturbed filepaths are in outpath
    config.subjects.subject_basepath = config.subjects.subject_outpath
    subject_list = read_subject_list(config["subjects"]["subject_list"])
    config["dataset"] = dataset
    if config["dataset"] != "verse":
        subject_list = subject_list.flatten()

    print(f"Number of subjects {len(subject_list)}")
    input_seg_fileformat = config.filename_convention.input.seg
    original_seg_path = f"{config.subjects.subject_basepath}/{input_seg_fileformat}"

    SEED = 12345
    train_subjects, test_subjects = train_test_split(
        subject_list, test_size=0.15, shuffle=True, random_state=SEED
    )
    train_subjects, val_subjects = train_test_split(
        train_subjects, test_size=0.15, shuffle=True, random_state=SEED
    )
    print(
        f"train {len(train_subjects)} val {len(val_subjects)} test {len(test_subjects)}"
    )

    train_paths = get_fullpaths(
        train_subjects, config, None, args.ap, args.lat, args.seg
    )
    val_paths = get_fullpaths(val_subjects, config, None, args.ap, args.lat, args.seg)
    train_val_paths = get_fullpaths(np.concatenate((train_subjects, val_subjects)), config, None, args.ap, args.lat, args.seg)  # type: ignore

    # take perturbation angles into account
    test_paths = {"ap": [], "lat": [], "seg": []}
    for angle in config["xray_pose"]["perturbation_angle"]:
        perturbation_paths = get_fullpaths(
            test_subjects, config, angle, args.ap, args.lat, args.seg
        )
        test_paths["ap"].extend(perturbation_paths["ap"])
        test_paths["lat"].extend(perturbation_paths["lat"])
        test_paths["seg"].extend(perturbation_paths["seg"])

    # write csv
    df_train = pd.DataFrame(data=train_paths)
    df_val = pd.DataFrame(data=val_paths)
    df_train_val = pd.DataFrame(data=train_val_paths)
    df_test = pd.DataFrame(data=test_paths)

    print(df_train.describe())
    print(df_val.describe())
    print(df_test.describe())
    print(df_train_val.describe())

    for df, suffix in zip(
        [df_train, df_val, df_test, df_train_val], ["train", "val", "test", "train+val"]
    ):
        df.to_csv(
            Path(config_path)
            .with_name(Path(config_path).stem + "_" + suffix)
            .with_suffix(".csv")
        )
