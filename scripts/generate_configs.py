import yaml

from XrayTo3DShape import update_multiple_key_values_in_nested_dict, config_template


ctpelvic_dataset = {
    "bone": "hip",
    "res": 1.0,
    "size": 288,
    "subject_basepath": "2D-3D-Reconstruction-Datasets/ctpelvic1k/raw/COLONOG/BIDS",
    "subject_list": "external/XrayTo3DPreprocess/workflow/ctpelvic1k/subjects/colonog_subjects.lst",
    "ct": "ct.nii.gz",
    "seg": "hip.nii.gz",
}

ctpelvic_hip = update_multiple_key_values_in_nested_dict(
    config_template, ctpelvic_dataset
)

with open("configs/full/ctpelvic_hip.yaml", "w") as f:
    yaml.dump(ctpelvic_hip, f)
