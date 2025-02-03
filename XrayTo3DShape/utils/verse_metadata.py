# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""query verse metadata"""
import math
import warnings
from typing import Dict

import pandas as pd

from .enums import DatasetConsts, VerseKeys
from .misc_utils import split_subject_vertebra_id


class VerseExcelSheet:
    """Reads the annotation excel sheet and allows to query vertebra level, shape, grade"""

    n_rows, n_cols = 161, 48
    data_frame: pd.DataFrame

    def __init__(
        self, annotations_filename="metadata/ryai190138_appendixe1.xlsx"
    ) -> None:
        warnings.filterwarnings("ignore", category=UserWarning)
        self.annotations_filename = annotations_filename
        # UserWarning: Unknown extension is not supported and will be removed
        self._read_extra_annotations()

        self.gradeid_to_grade_key = {
            "0": VerseKeys.NORMAL,
            "1": VerseKeys.MILD,
            "2": VerseKeys.MODERATE,
            "3": VerseKeys.SEVERE,
            "x": VerseKeys.FOREIGN_MATERIAL,
        }
        self.shapeid_to_shape_key = {
            "0": VerseKeys.NORMAL,
            "1": VerseKeys.WEDGE,
            "2": VerseKeys.BICONCAVE,
            "3": VerseKeys.CRUSH,
            "x": VerseKeys.FOREIGN_MATERIAL,
        }

    @classmethod
    def get_vertebra_keys(cls, filepath):
        """Input: /../../verse004_16.nii.gz"""
        subject_id, vertebra_id = split_subject_vertebra_id(filepath)
        return {VerseKeys.SUBJECT: subject_id, VerseKeys.VERTEBRA: vertebra_id}

    def has_foreign_material(self, vertebra_keys: Dict):
        """vertebra has cement, screws etc?"""
        return self.get_shape(vertebra_keys) == VerseKeys.FOREIGN_MATERIAL

    def get_shape(self, vertebra_keys: Dict) -> str:
        """Wedge, Concave, Crush, Normal"""
        if isinstance(vertebra_keys, str):
            vertebra_keys = self.get_vertebra_keys(vertebra_keys)
        vertebra_name = self.get_vertebra_name(vertebra_keys[VerseKeys.VERTEBRA])

        if vertebra_name.startswith("C"):
            return VerseKeys.NORMAL  # cervical vertebra do not have shape information
        column_name = self._get_shape_column_name(vertebra_name)
        shape_id = self._get_row_item(vertebra_keys, column_name)
        return self.shapeid_to_shape_key[self._cast_to_string(shape_id)]

    def get_severity(self, vertebra_keys: Dict) -> str:
        """Mild, Moderate, Severe"""
        vertebra_name = self.get_vertebra_name(vertebra_keys[VerseKeys.VERTEBRA])

        if vertebra_name.startswith("C"):
            return VerseKeys.NORMAL  # cervical vertebra do not have grade information
        column_name = self._get_grade_column_name(vertebra_name)
        grade_id = self._get_row_item(vertebra_keys, column_name)
        return self.gradeid_to_grade_key[self._cast_to_string(grade_id)]

    def get_ct_device(self, vertebra_keys: Dict):
        """Siemens, Toshiba etc."""
        return self._get_row_item(vertebra_keys, "CT_device")

    def get_ct_resolution(self, vertebra_keys: Dict):
        """resolution along axial direction"""
        return self._get_row_item(vertebra_keys, "Res")

    def get_bone_mass_density(self, vertebra_keys: Dict):
        """Bone Mass Density"""
        return self._get_row_item(vertebra_keys, "BMD")

    def get_vertebra_level(self, vertebra_keys: Dict):
        """Cervical, Thoracic, Lumbar"""
        vertebra_id = vertebra_keys[VerseKeys.VERTEBRA]

        if vertebra_id >= 1 and vertebra_id <= DatasetConsts.LAST_CERVICAL:
            return VerseKeys.CERVICAL

        if (
            vertebra_id > DatasetConsts.LAST_CERVICAL
            and vertebra_id <= DatasetConsts.LAST_THORACIC
        ):
            return VerseKeys.THORACIC

        if (
            vertebra_id > DatasetConsts.LAST_THORACIC
            and vertebra_id <= DatasetConsts.LAST_LUMBAR
        ):
            return VerseKeys.LUMBAR

        if vertebra_id == 25:
            return "others"
        # raise Exception(f'{vertebra_id} is an invalid Vertebra Number. 1-7 is cervical, 8-19 is thoracic. 20-24 is lumbar')

    def _cast_to_string(self, index) -> str:
        """pandas tries to convert cell values to appropriate type
        possible index values: 'x' 0, 0.0, NaN
        NaN can occur if the cell is empty
        """
        if isinstance(index, str) and index.startswith("x"):
            index = index
        elif isinstance(index, int):
            index = str(index)
        elif isinstance(index, float):
            index = str(0) if math.isnan(index) else str(int(index))
        else:
            index = index
            print(index)
        return index

    @staticmethod
    def get_vertebra_name(vertebra_number: int) -> str:
        """returns T1-T12, L1-L5, given vertebra number
        vertebra are numbered from 1-25. Thoracic range 8-19.Lumbar range 20-25
        >>> get_vertebra_name(5)
        'C5'
        >>> get_vertebra_name(10)
        'T3'
        >>> get_vertebra_name(25)
        'L6'
        """
        if vertebra_number >= 1 and vertebra_number <= 7:
            # cervical vertebra
            return f"C{vertebra_number}"
        elif vertebra_number >= 8 and vertebra_number <= 19:
            # Thoracic vertebra
            return f"T{vertebra_number - 7}"
        elif vertebra_number >= 20 and vertebra_number <= 25:
            # Lumbar vertebra
            return f"L{vertebra_number - 19}"
        else:
            raise ValueError(
                f"Vertebra number out-of-range. Recieved {vertebra_number}"
            )

    @staticmethod
    def _get_shape_column_name(vertebra_name: str):
        """
        >>> get_shape_column_name('T1')
        'T1_fx-s'
        """
        return f"{vertebra_name}_fx-s"

    @staticmethod
    def _get_grade_column_name(vertebra_name: str):
        """T1 -> T1_fx-g
        >>> get_grade_column_name('T1')
        'T1_fx-g'
        """
        return f"{vertebra_name}_fx-g"

    def _get_row_item(self, vertebra_keys: Dict, column_name):
        """get whole row"""
        subject_details = self._get_excel_row(vertebra_keys)
        return subject_details[column_name].values[0]

    def _get_excel_row(self, vertebra_keys):
        return self.data_frame.loc[
            self.data_frame.verse_ID == vertebra_keys[VerseKeys.SUBJECT]
        ]

    def _read_extra_annotations(self):
        """read metadata from xls"""
        # read annotations
        annotations = pd.read_excel(self.annotations_filename)

        # remove empty rows and columns
        self.data_frame = annotations.head(self.n_rows).iloc[:, : self.n_cols]
