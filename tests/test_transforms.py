
if __name__ == "__main__":
    from monai.transforms import *
    import numpy as np
    import matplotlib.pyplot as plt
    from XrayTo3DShape import *

    transforms = get_nonkasten_transforms()
    ap_transform, lat_transform, seg_transform = transforms['ap'], transforms['lat'], transforms['seg']

    ap_img_path = "2D-3D-Reconstruction-Datasets/LIDC-test/subjectwise/LIDC-0001/derivatives/xray_from_ct/LIDC-0001_vert-9_ap.png"
    lat_img_path = "2D-3D-Reconstruction-Datasets/LIDC-test/subjectwise/LIDC-0001/derivatives/xray_from_ct/LIDC-0001_vert-9_lat.png"
    seg_img_path = "2D-3D-Reconstruction-Datasets/LIDC-test/subjectwise/LIDC-0001/derivatives/seg_roi/LIDC-0001_vert-9-seg-vert_msk.nii.gz"

    ap_dict = ap_transform({"ap": ap_img_path})
    lat_dict = lat_transform({"lat": lat_img_path})
    seg_dict = seg_transform({"seg": seg_img_path})

    print(ap_dict['ap_meta_dict'])
    ap_img = ap_dict["ap"]
    lat_img = lat_dict["lat"]
    seg_img = seg_dict["seg"]

    print(ap_img.shape, lat_img.shape, seg_img.shape)

    fig = plt.figure(figsize=(4,4))
    plt.imshow(ap_img[0], cmap="gray")
    plt.axis("off")
    fig = plt.figure(figsize=(4,4))
    plt.axis("off")
    plt.imshow(lat_img[0], cmap="gray")

    seg_slices = get_projectionslices_from_3d(seg_img.squeeze())
    fig, axes = create_figure(*seg_slices)
    for ax, img in zip(axes, seg_slices):
        ax.imshow(img, cmap=plt.cm.gray)

    plt.show()