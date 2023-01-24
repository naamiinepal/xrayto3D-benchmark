# Dataset Notes

Any dataset will require AP, LAT and SEG image paths as input. Corresponding to these images TRANSFORMATIONS has to be defined that convert it to appropriate Tensor shape.

Currently, this is implemented as follows:

The `BaseDataset` takes list of `dicts` as input data. This make look as follows
```python
paths = [{'ap':'ap.png','lat':'lat.png','seg':'seg.nii.gz'} ]
```
The `transforms` may be defined as follows:
```python 
transforms = {'ap':ap_transform, 'lat':lat_transform, 'seg':seg_transform}
```
Each `transform` has to be a `compose` of `Dictionary`-based MONAI transformations.
```python
    lidc_seg_transform = Compose(
        [
            LoadImageD(
                keys={"seg"},
                ensure_channel_first=True,
                dtype=np.uint8,
                simple_keys=True,
                image_only=False,
            ),
            SpacingD(
                keys={"seg"},
                pixdim=(1, 1, 1),
                mode="nearest",
                padding_mode="zeros",
            ),
            SpatialPadD(keys={"seg"}, spatial_size=(96, 96, 96)),
            OrientationD(keys={"seg"}, axcodes="PIR"),
        ]
    )
```

## Rib Dataset (Original: TotalSegmentor)
Rib Dataset: 
- find views with all rib bones intact (reject cropped view)
- Combine individual rib segments into a single segmentation 
- Generate biplanar x-ray to 3D Segmentation dataset

Total Ribs samples : 481 

## Femur Dataset (Original: TotalSegmentor)
Femur Dataset:
-  Collect stats on femur segmentation volume
-  Choose all femur segmentation with volume higher than a threshold. (There are various cropped views of femur bone. We want to choose those with sufficient bone segment. Crop bones if the segmentation consists of large section of femur length. We only want femur head and few centimeters of the bone.)
-  Mirror the right femur so that it looks like a left femur.

Total Femur samples (threshold 50k voxels): 451

## Pelvic Dataset (Original: Totalsegmentor)
Pelvic Dataset:
- Find views with pelvic bones intact (not possible to reject cropped view)
- Generate biplanar x-ray to 3D segmentation dataset
  
Total Pelvic samples : 