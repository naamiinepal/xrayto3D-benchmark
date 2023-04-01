from XrayTo3DShape import VerseExcelSheet, split_subject_vertebra_id

metadata = VerseExcelSheet()
assert metadata.get_vertebra_name(11) == "T4"

subject_id, vertebra_id = split_subject_vertebra_id(
    "sub-verse061_22_seg-vert_msk.nii.gz"
)
assert subject_id == 61
assert vertebra_id == 22

subject_id, vertebra_id = split_subject_vertebra_id(
    "sub-verse410_split-verse227_vert-1-seg-vert_msk.nii.gz"
)
assert subject_id == 227
assert vertebra_id == 1
