
#----------------- Calculate Positive Class Weights for Cross Entropy Loss----#

'''
    when foreground and background voxels are imbalanced, the cross entropy loss
    may not be well calibrated. 
    Additional priors by providing predefined positive 
    class weight (w+) = #bg_voxels / #fg_voxels
    can be useful.

    TODO: unusual variation in w+ for femur. Investigate? w+ == 0? why
    Vertebra avg pos weight 19.21 +/- 4.08
    femur avg pos weight 13.75 +/- 339642.12
    rib avg pos weight 6435.80 +/- 12940.97
    hip avg pos weight 85.21 +/- 194.05
'''
import math
from pathlib import Path

import numpy as np
from xrayto3d_preprocess import get_segmentation_stats, read_image


def get_bg_fg_ratio(nifti_filename,verbose=False):
    seg = read_image(str(nifti_filename))
    stats = get_segmentation_stats(seg)
    fg_voxels = np.sum([stats.GetNumberOfPixels(label=l) for l in stats.GetLabels()])
    total_voxels = np.prod(seg.GetSize())
    bg_voxels = total_voxels - fg_voxels
    if fg_voxels == 0.0:
        pos_class_weight = float('nan')
    else:
        pos_class_weight = bg_voxels / fg_voxels
        
    if verbose:
        print(f'{pos_class_weight:.2f}')
    return pos_class_weight

lidc_path = '2D-3D-Reconstruction-Datasets/lidc/subjectwise/LIDC-IDRI-LUNA-0000/derivatives/seg_roi'

pos_weight_ratios = [ get_bg_fg_ratio(p) for p in Path(lidc_path).glob('*.nii.gz')]
print(f'LIDC {np.mean(pos_weight_ratios):.2f} +/- {np.std(pos_weight_ratios):.2f}')

totalseg_path = '2D-3D-Reconstruction-Datasets/totalsegmentator/Totalsegmentator_dataset'
pos_weight_ratios = [get_bg_fg_ratio(p,verbose=False) for id,p in enumerate(Path(totalseg_path).rglob('*femur*.nii.gz')) if id < 100]
print(f'Totalseg-femur {np.nanmedian(pos_weight_ratios):.2f} +/- {np.nanstd(pos_weight_ratios):.2f}')

pos_weight_ratios = [get_bg_fg_ratio(p,verbose=False) for id,p in enumerate(Path(totalseg_path).rglob('*rib*.nii.gz')) if id < 100]
print(f'Totalseg-rib {np.nanmedian(pos_weight_ratios):.2f} +/- {np.nanstd(pos_weight_ratios):.2f}')

pos_weight_ratios = [get_bg_fg_ratio(p,verbose=False) for id,p in enumerate(Path(totalseg_path).rglob('*hip*.nii.gz')) if id < 100]
print(f'Totalseg-hip {np.nanmedian(pos_weight_ratios):.2f} +/- {np.nanstd(pos_weight_ratios):.2f}')