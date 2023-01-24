from xrayto3d_preprocess import *

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')

    args = parser.parse_args()

    # config_path = "configs/test/Verse2020-DRR-test.yaml"
    config_path = args.config_file
    # use config to find the fullpath of x-ray and seg_roi pairs

    config = read_config_and_load_components(config_path)
    subject_list = read_subject_list(config['subjects']['subject_list']).flatten()

    print(subject_list)
    input_seg_fileformat = config.filename_convention.input.seg
    original_seg_path = f"{config.subjects.subject_basepath}/{input_seg_fileformat}"

    ap_paths,lat_paths,seg_paths = [],[],[]
    for subject_id in subject_list:
        derivatives_path = (
            Path(config.subjects.subject_basepath).resolve() / f"{subject_id}" / "derivatives"
        )
        xray_basepath = derivatives_path / "xray_from_ct"
        seg_roi_basepath = derivatives_path / "seg_roi"

        vert_xray_ap = sorted(xray_basepath.rglob("*ap.png"))
        vert_xray_lat = sorted(xray_basepath.rglob("*lat.png"))
        vert_seg = sorted(seg_roi_basepath.rglob("*msk.nii.gz"))

        ap_paths.extend(vert_xray_ap)
        lat_paths.extend(vert_xray_lat)
        seg_paths.extend(vert_seg)
    
    # write csv
    dicts_path = {'ap':ap_paths, 'lat':lat_paths,'seg': seg_paths}
    df = pd.DataFrame(data=dicts_path)

    df.to_csv(Path(config_path).with_suffix('.csv'))