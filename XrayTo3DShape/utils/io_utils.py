from pathlib import Path
import SimpleITK as sitk

def write_image(img, out_path,pixeltype=None):
    if isinstance(out_path, Path):
        out_path = str(out_path)
    if pixeltype:
        img = sitk.Cast(img,pixeltype)
    sitk.WriteImage(img, out_path)

def read_image(img_path):
    """returns the SimpleITK image read from given path

    Parameters:
    -----------
    pixeltype (ImagePixelType):
    """

    if isinstance(img_path, Path):
        img_path = str(img_path)

    # if pixeltype == ImagePixelType.ImageType:
    #     pixeltype = sitk.sitkUInt16
    #     return sitk.ReadImage(img_path,pixeltype)

    # elif pixeltype == ImagePixelType.SegmentationType:
    #     pixeltype = sitk.sitkUInt8
    #     return sitk.ReadImage(img_path,pixeltype)

    # else:
    #     raise ValueError(f'ImagePixelType cannot be {pixeltype}')

    return sitk.ReadImage(img_path)