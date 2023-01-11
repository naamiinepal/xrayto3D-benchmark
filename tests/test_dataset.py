from XrayTo3DShape.datasets.base_dataset import BaseDataset
from XrayTo3DShape.datasets.transforms import get_kasten_transforms

if __name__ == '__main__':
    import pandas as pd

    paths_location = 'configs/test/LIDC-DRR-test.csv'
    paths = pd.read_csv(paths_location,index_col=0).to_numpy()
    paths = [ {'ap':ap,'lat':lat,'seg':seg} for ap,lat,seg in paths] 

    ds = BaseDataset(data=paths,transforms=get_kasten_transforms())
    ap,lat,seg = ds[0]
    print(ap['ap'].shape,lat['lat'].shape,seg['seg'].shape)