from pathlib import Path
from typing import Sequence
from xrayto3d_preprocess import load_centroids
import numpy as np

def generate_augment_path(sub_dir:str, name:str, subject_id, vert_id, augment_angle, output_path_template, config):
    """xray_ap:"{id_hip-ap.png} -> img001_hip-ap.png"""
    output_fileformat = config['filename_convention']['output']
    out_dirs = config['out_directories']
    filename = output_fileformat[name].format(id=subject_id,vert=vert_id,angle=augment_angle)
    out_path = output_path_template.format(output_type=out_dirs[sub_dir],output_name=filename)
    Path(out_path).parent.mkdir(exist_ok=True, parents=True)
    return out_path

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


def get_individual_fullpaths_aug(
    subject_dir,
    config,
):
    """return ap, lat, seg filepaths"""
    subject_id = subject_dir
    subject_basepath = config['subjects']['subject_basepath']
    input_fileformat = config['filename_convention']['input']

    centroid_path = Path(subject_basepath)/subject_id/input_fileformat['ctd'].format(id=subject_id)
    _, centroids = load_centroids(centroid_path)

    OUT_PATH_TEMPLATE = f'{subject_basepath}/{subject_id}/{config["out_directories"]["derivatives"]}/{{output_type}}/{{output_name}}'

    xray_ap = []
    xray_lat = []
    seg = []
    for vert_id, *ctd in centroids:
        for augment_angle in range(0,360,5):
            out_seg_aug_path = generate_augment_path('seg_roi_augment','vert_seg_aug',subject_id,vert_id,augment_angle,output_path_template=OUT_PATH_TEMPLATE,config=config)

            out_xray_ap_path = generate_augment_path('xray_from_ct_augment','vert_xray_aug_ap',subject_id,vert_id,augment_angle,OUT_PATH_TEMPLATE,config)

            out_xray_lat_path = generate_augment_path('xray_from_ct_augment','vert_xray_aug_lat',subject_id, vert_id, augment_angle, OUT_PATH_TEMPLATE, config)

            xray_ap.append(out_xray_ap_path)
            xray_lat.append(out_xray_lat_path)
            seg.append(out_seg_aug_path)

    xray_ap = sorted(xray_ap)
    xray_lat = sorted(xray_lat)
    seg = sorted(seg)
    return xray_ap, xray_lat, seg

def get_fullpaths(subject_list: Sequence, config, ap_pattern, lat_pattern, seg_pattern,aug_based=True):
    """return ap, lat, seg paths as dict"""
    print(ap_pattern, lat_pattern, seg_pattern)
    ap, lat, seg = [], [], []
    if config["dataset"] == "verse":
        for subject, subject_dir in subject_list:
            if aug_based:
                _ap, _lat, _seg = get_individual_fullpaths_aug(
                subject_dir, config)
            else:
                _ap, _lat,_seg = get_individual_fullpaths(subject_dir,config,ap_pattern,lat_pattern,seg_pattern,False)
            ap.extend(_ap)
            lat.extend(_lat)
            seg.extend(_seg)
    else:
        for subject in subject_list:
            if aug_based:
                _ap, _lat, _seg = get_individual_fullpaths_aug(
                subject, config)
            else:
                _ap, _lat, _seg = get_individual_fullpaths(subject,config,ap_pattern,lat_pattern,seg_pattern,False)
            ap.extend(_ap)
            lat.extend(_lat)
            seg.extend(_seg)
    assert len(ap) == len(
        seg
    ), f"Number of segmentations {len(seg)} and Number of AP xrays {len(ap)} do not match"

    return {"ap": ap, "lat": lat, "seg": seg}

if __name__ == '__main__':
    import argparse
    import pandas as pd
    from tqdm import tqdm
    from sklearn.model_selection import train_test_split
    from xrayto3d_preprocess import read_config_and_load_components, read_subject_list

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')
    parser.add_argument('--anatomy')
    parser.add_argument('--debug',default=False,action='store_true')

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


    config = read_config_and_load_components(args.config_file)
    config_path = args.config_file
    print(config)

    subject_list = read_subject_list(config['subjects']['subject_list'])
    config["dataset"] = dataset
    if config["dataset"] != "verse":
        subject_list = subject_list.flatten()
    print(f"Number of subjects {len(subject_list)}")

    if args.debug:
        subject_list = subject_list[:4]

    # split subjects into train/val/test
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
    train_paths = get_fullpaths(train_subjects, config, args.ap, args.lat, args.seg,aug_based=True)
    train_noaug_paths = get_fullpaths(train_subjects,config,args.ap,args.lat,args.seg,aug_based=False)
    val_paths = get_fullpaths(val_subjects, config, args.ap, args.lat, args.seg,aug_based=False)
    train_val_paths = get_fullpaths(np.concatenate((train_subjects, val_subjects)), config, args.ap, args.lat, args.seg,aug_based=True)  # type: ignore
    test_paths = get_fullpaths(test_subjects, config, args.ap, args.lat, args.seg,aug_based=False)
    # write csv
    df_train = pd.DataFrame(data=train_paths)
    df_train_noaug = pd.DataFrame(data=train_noaug_paths)
    df_val = pd.DataFrame(data=val_paths)
    df_train_val = pd.DataFrame(data=train_val_paths)
    df_test = pd.DataFrame(data=test_paths)

    print(df_train.describe())
    print(df_val.describe())
    print(df_test.describe())
    print(df_train_val.describe())

    for df, suffix in zip(
        [df_train,df_train_noaug, df_val, df_test, df_train_val], ["train", "train_noaug", "val", "test", "train+val"]
    ):
        df.to_csv(
            Path(config_path)
            .with_name(Path(config_path).stem + "_" + suffix)
            .with_suffix(".csv")
        )
