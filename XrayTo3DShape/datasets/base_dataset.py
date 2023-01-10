from torch.utils.data import Dataset
from monai.transforms.transform import apply_transform
from typing import Sequence, Callable, Dict, Any


class BaseDataset(Dataset):
    """
    A generic dataset with a length property and required callable data transform for AP, LAT and segmentation  when fetching a data sample.
            [{
             'ap': 'image1.nii.gz', 'lat':'image3.nii.gz','seg': 'label1.nii.gz'
             }
            ]

    """

    def __init__(self, data: Sequence, transforms: Dict[str, Callable]) -> None:
        super().__init__()
        self.data = data
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`
        """
        data_i = self.data[index]
        ap_transform, lat_transform, seg_transform = (
            self.transforms["ap"],
            self.transforms["lat"],
            self.transforms["seg"],
        )
        return (
            apply_transform(ap_transform, data_i),
            apply_transform(lat_transform, data_i),
            apply_transform(seg_transform, data_i),
        )

    def __getitem__(self, index):
        return self._transform(index)
