from XrayTo3DShape import (
    AtlasDeformationDataset,
    BaseDataset,
    get_atlas_deformation_transforms,
    get_kasten_transforms,
)

if __name__ == "__main__":
    import pandas as pd

    paths_location = "configs/test/LIDC-DRR-test.csv"
    paths = pd.read_csv(paths_location, index_col=0).to_numpy()
    paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]

    ds = BaseDataset(data=paths, transforms=get_kasten_transforms())
    ap, lat, seg = ds[0]
    print(ap["ap"].shape, lat["lat"].shape, seg["seg"].shape)

    atlas_path = "2D-3D-Reconstruction-Datasets/lidc/subjectwise/LIDC-IDRI-LUNA-0001/derivatives/seg_roi/LIDC-IDRI-LUNA-0001_vert-15-seg-vert_msk.nii.gz"

    atlas_ds = AtlasDeformationDataset(
        data=paths, transforms=get_atlas_deformation_transforms(), atlas_path=atlas_path
    )

    ap, lat, seg, atlas = atlas_ds[0]

    print(ap["ap"].shape, lat["lat"].shape, seg["seg"].shape, atlas["atlas"].shape)
