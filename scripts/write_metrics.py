from pathlib import Path
import csv
from monai.metrics.hausdorff_distance import HausdorffDistanceMetric
from monai.metrics.meandice import DiceMetric
from monai.metrics.surface_distance import SurfaceDistanceMetric
import numpy as np
from surface_distance.metrics import compute_surface_distances, compute_surface_overlap_at_tolerance
from XrayTo3DShape import read_image, to_numpy
import SimpleITK as sitk
import torch 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()


target_dir = f'{args.path}/combined_patches'

gt_paths = sorted(list(Path(target_dir).glob('*rib_msk_gt.nii.gz')))
pred_paths = sorted(list(Path(target_dir).glob('*rib_msk_pred.nii.gz')))

print(f'gt {len(gt_paths)} pred {len(pred_paths)}')

def get_subject_id(filename,PREFIX_LEN = len('s0046')):
    return filename.name[:PREFIX_LEN]
    
with open(Path(target_dir) / 'metric-log.csv', 'w') as f:
    filestream_writer = csv.writer(f)


    header = ['subject-id','DSC','ASD','HD95','NSD']
    filestream_writer.writerow(header)

    # metric callables
    VOXEL_SPACING = (1.25,)*3
    NSD_TOLERANCE = 1.5
    DSC = DiceMetric()
    ASD = SurfaceDistanceMetric()
    HD95 = HausdorffDistanceMetric(percentile=95)
    NSD = lambda gt, pred: compute_surface_overlap_at_tolerance(compute_surface_distances(
        gt.astype(bool), pred.astype(bool), VOXEL_SPACING)
    ,NSD_TOLERANCE)[0]

    for gt_p, pred_p in zip(gt_paths, pred_paths):
        gt_subject, pred_subject = get_subject_id(gt_p), get_subject_id(pred_p)
        

        assert gt_subject == pred_subject, f'expected {gt_subject} and {pred_subject} to be same'

        subject_id = gt_subject

        gt:np.ndarray = sitk.GetArrayFromImage(read_image(str(gt_p)))
        pred:np.ndarray = sitk.GetArrayFromImage(read_image(str(pred_p)))

        

        gt = gt[np.newaxis,np.newaxis,...]
        pred = pred[np.newaxis,np.newaxis,...]
        dsc = DSC(torch.from_numpy(gt), 
                  torch.from_numpy(pred))
        dsc = to_numpy(dsc).flatten().tolist()[0]

        asd = ASD(torch.from_numpy(gt),
                  torch.from_numpy(pred))
        asd = to_numpy(asd).flatten().tolist()[0]

        hd95 = HD95(torch.from_numpy(gt),
                    torch.from_numpy(pred))
        hd95 = to_numpy(hd95).flatten().tolist()[0]

        nsd = NSD(gt[0,0],
                  pred[0,0])
        # print(f'{subject_id} shape {gt.shape} DSC {dsc} ASD {asd} HD95 {hd95} NSD {nsd}' )
        filestream_writer.writerow([subject_id, f'{dsc:.2f}', f'{asd:.2f}', f'{hd95:.2f}', f'{nsd:.2f}'])

