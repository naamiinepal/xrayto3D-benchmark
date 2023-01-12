import pandas as pd
from XrayTo3DShape import *
from torch.utils.data import DataLoader
from monai.losses.dice import DiceCELoss
import torch
from monai.metrics.meandice import DiceMetric
from monai.networks.nets.unet import UNet
from monai.transforms import *
import wandb
from tqdm import trange

lr = 1e-2
NUM_EPOCHS = 1000
wandb.init(project="pipeline-test-01", name="denoising-autoencoder-01")

paths_location = "configs/test/LIDC-DRR-test.csv"
paths = pd.read_csv(paths_location, index_col=0).to_numpy()
paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]

ds = BaseDataset(data=paths, transforms=get_denoising_autoencoder_transforms(size=64, resolution=1.5))
data_loader = DataLoader(ds, batch_size=1)
ap, lat, seg = next(iter(data_loader))
print(seg['orig'].shape,seg['gaus'].shape)

def train(dict_key_for_training, max_epochs=10, learning_rate=1e-3):

    model = AutoEncoder1DEmbed(
        spatial_dims=3,
        in_shape = (1,64,64,64),
        out_channels=1,
        latent_size= 64,
        channels=(96,256,384,256),
        strides=(2, 2, 2, 2),
    )

    # Create loss fn and optimiser
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    epoch_loss_values = []

    t = trange(
        max_epochs,
        desc=f"{dict_key_for_training} -- epoch 0, avg loss: inf", leave=True)
    for epoch in t:
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in data_loader:
            step += 1
            ap, lat, seg = batch_data
            inputs = seg[dict_key_for_training]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, seg['orig'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            wandb.log({'loss':loss.item()})
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        t.set_description(
            f"{dict_key_for_training} -- epoch {epoch + 1}"
            + f", average loss: {epoch_loss:.4f}")
    return model, epoch_loss_values

model, epoch_loss_vals = train('gaus',learning_rate=lr,max_epochs=100)
torch.save(model,'denoising_ae_epoch100.pth')