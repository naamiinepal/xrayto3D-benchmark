"""generate data sample filepaths split into train, val, test sets"""
from pathlib import Path
from typing import Sequence
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xrayto3d_preprocess import read_config_and_load_components, read_subject_list


def get_individual_fullpaths(
    subject_dir,
    config,
    ap_pattern="*ap.png",
    lat_pattern="*lat.png",
    seg_pattern="*msk.nii.gz",
    patch_based=False
):
    """return ap, lat, seg filepaths"""
    derivatives_path = (
        Path(config.subjects.subject_basepath).resolve()
        / f"{subject_dir}"
        / "derivatives"
    )
    if patch_based:
        xray_basepath = derivatives_path / "xray_from_ct_patch"
        seg_roi_basepath = derivatives_path / "seg_roi_patch"
    else:
        xray_basepath = derivatives_path / "xray_from_ct"
        seg_roi_basepath = derivatives_path / "seg_roi"

    xray_ap = sorted(xray_basepath.rglob(ap_pattern))
    xray_lat = sorted(xray_basepath.rglob(lat_pattern))
    seg = sorted(seg_roi_basepath.rglob(seg_pattern))
    return xray_ap, xray_lat, seg


def get_fullpaths(subject_list: Sequence, config, ap_pattern, lat_pattern, seg_pattern, patch_based=False):
    """return ap, lat, seg paths as dict"""
    print(ap_pattern, lat_pattern, seg_pattern)
    ap, lat, seg = [], [], []
    if config["dataset"] == "verse":
        for subject, subject_dir in subject_list:
            _ap, _lat, _seg = get_individual_fullpaths(
                subject_dir, config, ap_pattern, lat_pattern, seg_pattern, patch_based
            )
            ap.extend(_ap)
            lat.extend(_lat)
            seg.extend(_seg)
    else:
        for subject in subject_list:
            _ap, _lat, _seg = get_individual_fullpaths(
                subject, config, ap_pattern, lat_pattern, seg_pattern, patch_based
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
    parser.add_argument('--anatomy')
    # parser.add_argument("--ap", type=str, default="*hip-ap.png")
    # parser.add_argument("--lat", type=str, default="*hip-lat.png")
    # parser.add_argument("--seg", type=str, default="*hip_msk.nii.gz")
    # parser.add_argument("--ap", type=str, default="*vert*_ap.png")
    # parser.add_argument("--lat", type=str, default="*vert*_lat.png")
    # parser.add_argument("--seg", type=str, default="*vert_msk.nii.gz")
    parser.add_argument('--patch',default=False,action='store_true')

    args = parser.parse_args()
    print(args)

    if args.anatomy == 'hip':
        args.ap = '*hip-ap.png'
        args.lat = '*hip-lat.png'
        args.seg = '*hip_msk.nii.gz'
    elif args.anatomy == 'rib':
        args.ap = '*rib-ap.png'
        args.lat = '*rib-lat.png'
        args.seg = '*rib_msk.nii.gz'
    elif args.anatomy == 'vertebra':
        args.ap = '*ap.png'
        args.lat = '*lat.png'
        args.seg = '*msk.nii.gz'
    elif args.anatomy == 'femur':
        args.ap = '*femur*-ap.png'
        args.lat = '*femur*-lat.png'
        args.seg = '*femur*_msk.nii.gz'
    else:
        raise ValueError(f'anatomy {args.anatomy} is not valid. Expected one of hip, femur, vertebra, rib')
    
    dataset = (
        "verse"
        if str(args.config_file).split("-")[0].lower().startswith("verse")
        else "others"
    )

    # config_path = "configs/test/Verse2020-DRR-test.yaml"
    config_path = args.config_file
    # use config to find the fullpath of x-ray and seg_roi pairs

    config = read_config_and_load_components(config_path)
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

    train_paths = get_fullpaths(train_subjects, config, args.ap, args.lat, args.seg,patch_based=args.patch)
    val_paths = get_fullpaths(val_subjects, config, args.ap, args.lat, args.seg,patch_based=args.patch)
    train_val_paths = get_fullpaths(np.concatenate((train_subjects, val_subjects)), config, args.ap, args.lat, args.seg, patch_based=args.patch)  # type: ignore
    test_paths = get_fullpaths(test_subjects, config, args.ap, args.lat, args.seg, patch_based=args.patch)
    # write csv
    df_train = pd.DataFrame(data=train_paths)
    df_val = pd.DataFrame(data=val_paths)
    df_train_val = pd.DataFrame(data=train_val_paths)
    df_test = pd.DataFrame(data=test_paths)

    print(df_train.describe())
    print(df_val.describe())
    print(df_test.describe())
    print(df_train_val.describe())

    patch_based_suffix = '_patch' if args.patch else ''
    for df, suffix in zip(
        [df_train, df_val, df_test, df_train_val], ["train", "val", "test", "train+val"]
    ):
        df.to_csv(
            Path(config_path)
            .with_name(Path(config_path).stem + "_" + suffix + patch_based_suffix)
            .with_suffix(".csv")
        )
