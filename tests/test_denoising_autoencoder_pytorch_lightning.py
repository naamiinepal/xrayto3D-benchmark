import pytorch_lightning as pl
from typing import Dict, Any
from monai.networks.nets.autoencoder import AutoEncoder
import torch
from XrayTo3DShape import *
import pandas as pd
from torch.utils.data import DataLoader

class DenoisingAETrainer(pl.LightningModule):
    def __init__(self, config: Dict) -> None:
        self.config = config
        self.model = AutoEncoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=(4, 8, 16, 32),
            strides=(2, 2, 2, 2),
        )

        paths_location = "configs/test/LIDC-DRR-test.csv"
        paths = pd.read_csv(paths_location, index_col=0).to_numpy()
        self.paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]
    def configure_optimizers(self) -> Any:
        return torch.optim.AdamW(self.parameters(), lr= 1e-2)
    
    def train_dataloader(self):
        ds = BaseDataset(data = self.paths,transforms=get_denoising_autoencoder_transforms(size=64,resolution=1.5))
        return DataLoader(ds, batch_size=1)
