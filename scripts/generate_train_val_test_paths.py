from xrayto3d_preprocess import *
from sklearn.model_selection import train_test_split

def get_individual_fullpaths(subject_dir,config):
    derivatives_path = (
            Path(config.subjects.subject_basepath).resolve() / f"{subject_dir}" / "derivatives"
        )
    xray_basepath = derivatives_path / "xray_from_ct"
    seg_roi_basepath = derivatives_path / "seg_roi"

    vert_xray_ap = sorted(xray_basepath.rglob("*ap.png"))
    vert_xray_lat = sorted(xray_basepath.rglob("*lat.png"))
    vert_seg = sorted(seg_roi_basepath.rglob("*msk.nii.gz"))
    return vert_xray_ap,vert_xray_lat,vert_seg

def get_fullpaths(subject_list:Sequence,config):
    ap,lat,seg = [], [], []
    if config['dataset'] == 'verse':
        for subject,subject_dir in subject_list:
            _ap,_lat,_seg = get_individual_fullpaths(subject_dir,config)
            ap.extend(_ap)
            lat.extend(_lat)
            seg.extend(_seg)
    else:
        for subject in subject_list:
            _ap,_lat,_seg = get_individual_fullpaths(subject,config)
            ap.extend(_ap)
            lat.extend(_lat)
            seg.extend(_seg)        
    assert len(ap) == len(seg), f'Number of segmentations {len(seg)} and Number of AP xrays {len(ap)} do not match'

    return {'ap':ap,'lat':lat,'seg':seg}

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file')

    args = parser.parse_args()

    dataset = 'verse' if str(args.config_file).split('-')[0].lower().startswith('verse') else 'others'

    # config_path = "configs/test/Verse2020-DRR-test.yaml"
    config_path = args.config_file
    # use config to find the fullpath of x-ray and seg_roi pairs

    config = read_config_and_load_components(config_path)
    subject_list = read_subject_list(config['subjects']['subject_list'])
    config['dataset'] = dataset
    if config['dataset'] != 'verse':
        subject_list = subject_list.flatten()

    print(f'Number of subjects {len(subject_list)}')
    input_seg_fileformat = config.filename_convention.input.seg
    original_seg_path = f"{config.subjects.subject_basepath}/{input_seg_fileformat}"

    SEED=12345
    train_subjects, test_subjects = train_test_split(subject_list,test_size=0.15,shuffle=True,random_state=SEED)
    train_subjects, val_subjects = train_test_split(train_subjects,test_size=0.15,shuffle=True,random_state=SEED)
    print(f'train {len(train_subjects)} val {len(val_subjects)} test {len(test_subjects)}')

    train_paths = get_fullpaths(train_subjects,config)
    val_paths = get_fullpaths(val_subjects,config)
    test_paths = get_fullpaths(test_subjects,config)
    # write csv
    df_train = pd.DataFrame(data=train_paths)
    df_val = pd.DataFrame(data=val_paths)
    df_test = pd.DataFrame(data=test_paths)

    print(df_train.describe())
    print(df_val.describe())
    print(df_test.describe())


    for df, suffix in zip([df_train,df_val,df_test],['train','val','test']):
        df.to_csv(Path(config_path).with_name(Path(config_path).stem+'_'+suffix).with_suffix('.csv'))
