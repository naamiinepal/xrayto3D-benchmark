from enum import Enum


class DatasetExclusion(Enum):
    EXCLUDE_CERVICAL = "exclude_cervical"
    EXCLUDE_FOREIGN = "exclude_foreign"


class VertebraType(Enum):
    CERVICAL = 0
    THORACIC = 1
    LUMBAR = 2


class DataExcelSheetKeys:
    SUBJECT_ID = "subject_ID"
    CT_DEVICE = "CT-Device"
    CT_RESOLUTION = "CT-Resolution"
    BMD = "BMD"


class DatasetType(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DatasetConsts:
    TRAIN_DIR = "train"
    VAL_DIR = "val"
    TEST_DIR = "test"
    IMAGE_PATTERN = "*.png"
    LABEL_PATTERN = "*.nii.gz"

    AP_KEY = "ap"
    LAT_KEY = "lat"
    IMG_KEY = "image"
    LABEL_KEY = "label"
    NOISY_LABEL_KEY = "noisy_label"
    CLEAN_LABEL_KEY = "clean_label"

    NUM_VERTEBRA_LEVELS = 24
    LAST_CERVICAL = 7
    LAST_THORACIC = 19
    LAST_LUMBAR = 24


class VerseKeys:
    CERVICAL = "cervical"
    THORACIC = "thoracic"
    LUMBAR = "lumbar"

    NORMAL = "normal"
    BICONCAVE = "biconcave"
    WEDGE = "wedge"
    CRUSH = "crush"

    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

    FOREIGN_MATERIAL = "foreign-material"

    SUBJECT = "subject"
    VERTEBRA = "vertebra"
