from synapseclient import Project
from xrayto3d_preprocess import rotate_about_image_center,generate_xray,generate_perturbed_xray,get_interpolator
import SimpleITK as sitk
from XrayTo3DShape.utils.io_utils import read_image, write_image
from xrayto3d_preprocess.enumutils import ProjectionType

rot_angle = 30

sample_ct_path = 'test_data/sub-verse004/derivatives/ct_roi/sub-verse004_vert-16_ct.nii.gz'
sample_seg_path = 'test_data/sub-verse004/derivatives/seg_roi/sub-verse004_vert-16-seg-vert_msk.nii.gz'
out_ct_path = f'tests/sub-verse004_vert-16_angle-{rot_angle}_ct.nii.gz'
out_seg_path = f'tests/sub-verse004_vert-16_angle-{rot_angle}_seg-vert_msk.nii.gz'
rot_out_xray_ap_path = f'tests/sub-verse004_vert-16_angle-{rot_angle}_ap.png'
rot_out_xray_lat_path = f'tests/sub-verse004_vert-16_angle-{rot_angle}_lat.png'
out_xray_ap_path = f'tests/sub-verse004_vert-16_ap.png'
out_xray_lat_path = f'tests/sub-verse004_vert-16_lat.png'

ct = read_image(sample_ct_path)
rotated_ct = rotate_about_image_center(ct,rx=0,ry=0,rz=360 - rot_angle,interpolator=get_interpolator('linear'))
write_image(rotated_ct,out_ct_path,pixeltype=ct.GetPixelID())

seg = read_image(sample_seg_path)
rotated_seg = rotate_about_image_center(seg,rx=0,ry=0,rz=360-rot_angle,interpolator=get_interpolator('nearest'))
write_image(rotated_seg,out_seg_path)

print(seg.GetDirection(),rotated_seg.GetDirection())
config = {
'ap':
    {
    'rx': -90,
    'ry': 0,
    'rz': 90    ,
    },
'lat':
    {
    'rx': -90,
    'ry': 0,
    'rz': 0,
    },
  'res': 1.0,
  'size': 96,
}
generate_xray(out_ct_path,ProjectionType.AP,None,config,rot_out_xray_ap_path)
generate_xray(out_ct_path,ProjectionType.LAT,None,config,rot_out_xray_lat_path)

generate_perturbed_xray(sample_ct_path,ProjectionType.AP,config,out_xray_ap_path,
                        rot_angle)
generate_perturbed_xray(sample_ct_path,ProjectionType.LAT,config,out_xray_lat_path,rot_angle)


# visualize correspondence

import matplotlib.pyplot as plt

from XrayTo3DShape import get_nonkasten_transforms, get_kasten_transforms,get_projectionslices_from_3d, create_figure

def save_projections(transforms):
    ap_transform, lat_transform, seg_transform = (
        transforms["ap"],
        transforms["lat"],
        transforms["seg"],
    )

    ap_img_path = out_xray_ap_path
    lat_img_path = out_xray_lat_path
    seg_img_path = out_seg_path

    ap_dict = ap_transform({"ap": ap_img_path})
    lat_dict = lat_transform({"lat": lat_img_path})
    seg_dict = seg_transform({"seg": seg_img_path})

    print(ap_dict["ap_meta_dict"])
    ap_img = ap_dict["ap"]
    if len(ap_img.shape) == 3: ap_img = ap_img[0] 
    elif len(ap_img.shape) == 4 : ap_img = ap_img[0,0]
    lat_img = lat_dict["lat"]
    if len(lat_img.shape) == 3: lat_img = lat_img[0] 
    elif len(lat_img.shape) == 4 : lat_img = lat_img[0,:,:,0]
    seg_img = seg_dict["seg"]

    print(ap_img.shape, lat_img.shape, seg_img.shape)

    fig = plt.figure(figsize=(4, 4))
    plt.imshow(ap_img, cmap="gray")
    plt.axis("off")
    fig = plt.figure(figsize=(4, 4))
    plt.axis("off")
    plt.imshow(lat_img, cmap="gray")

    seg_slices = get_projectionslices_from_3d(seg_img.squeeze())
    fig, axes = create_figure(*seg_slices,ap_img,lat_img)
    for ax, img in zip(axes, [*seg_slices,ap_img,lat_img]):
        ax.imshow(img, cmap=plt.cm.gray)

transforms = get_nonkasten_transforms()
save_projections(transforms)
plt.savefig('tests/transforms_360_augment.png')
plt.close()

transforms = get_kasten_transforms()
save_projections(transforms)
plt.savefig('tests/transforms_360_augment_kasten.png')
plt.close()

ap_img_path = out_xray_ap_path = '2D-3D-Reconstruction-Datasets/verse19/subjectwise/sub-verse004/derivatives/xray_from_ct_augment/sub-verse004_vert-16_angle-30_ap.png'
lat_img_path = out_xray_lat_path = '2D-3D-Reconstruction-Datasets/verse19/subjectwise/sub-verse004/derivatives/xray_from_ct_augment/sub-verse004_vert-16_angle-30_lat.png'
seg_img_path = out_seg_path = '2D-3D-Reconstruction-Datasets/verse19/subjectwise/sub-verse004/derivatives/seg_roi_augment/sub-verse004_vert-16_angle-30-seg-vert_msk.nii.gz'

transforms = get_nonkasten_transforms()
save_projections(transforms)
plt.savefig('tests/transforms_360_augment_2.png')
plt.close()

transforms = get_kasten_transforms()
save_projections(transforms)
plt.savefig('tests/transforms_360_augment_kasten_2.png')
plt.close()
