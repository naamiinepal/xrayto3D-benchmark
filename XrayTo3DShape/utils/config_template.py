"""
template for config file
id: sample id of each ct scan
bone: one of [femur, hip, ribs, vertebra]
res: voxel resolution
size: dimension of the cube representing the image
TODO: check res and size agree
"""

config_template = {
    "bone": "",
    "subjects": {
        "subject_basepath": "",
        "subject_list": "",
    },  # fill with path to file with ap, lat, seg paths
    "xray_pose": {
        "_load": "xray_pose_conf/${ROI_properties.axcode}_pose.yaml",  # read additional details from other places
        "res": "${ROI_properties.res}",
        "size": "${ROI_properties.size}",
        "drr_from_mask": "${ROI_properties.drr_from_mask}",
    },
    "out_directories": {"_load": "directory_conf/dir_ct.yaml"},
    "ROI_properties": {
        "axcode": "PIR",
        "extraction_ratio": {
            "L": 0.5,
            "A": 0.5,
            "S": 0.5,
        },  # crop around center of mass
        "drr_from_mask": False,
        "ct_padding": -1024,
        "seg_padding": 0,
        "res": 0,  # placeholder
        "size": 0,  # placeholder
    },
    "filename_convention": {
        "input": {
            "ct": "ct.nii.gz",
            "seg": "seg.nii.gz",
            "ctd": "",  # only used for vertebra body
        },
        "output": {
            "xray_ap": "{id}_${bone}_ap.png",
            "xray_lat": "{id}_lat.png",
            "xray_mask_ap": "{id}_mask-ap.png",
            "xray_mask_lat": "{id}_mask-lat.png",
            "ct_roi": "{id}_ct.nii.gz",
            "ct_mask_roi": "{id}_ct-mask.nii.gz",
            "seg_roi": "{id}_msk.nii.gz",
        },
    },
}
