import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import trange

import wandb
from XrayTo3DShape import (
    BaseDataset,
    CustomAutoEncoder,
    get_denoising_autoencoder_transforms,
    l1_loss,
    printarr,
)


def train(model, dict_key_for_training, max_epochs=10, learning_rate=1e-3):
    # Create loss fn and optimiser
    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), learning_rate)

    epoch_loss_values = []

    t = trange(
        max_epochs,
        desc=f"{dict_key_for_training} -- epoch 0, avg loss: inf",
        leave=True,
    )
    for epoch in t:
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in data_loader:
            step += 1
            ap, lat, seg = batch_data
            batch_input = seg[dict_key_for_training]
            optimizer.zero_grad()
            outputs, embedding_vec = model.forward(batch_input)
            sparsity_loss = l1_loss(embedding_vec)  # l1 regularization
            loss = loss_function(outputs, seg["orig"]) + 0.1 * sparsity_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if WANDB_ON:
                wandb.log({"loss": loss.item(), "sparsity_loss": sparsity_loss.item()})
            print("loss", loss.item(), "sparsity_loss", sparsity_loss.item())
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        t.set_description(
            f"{dict_key_for_training} -- epoch {epoch + 1}"
            + f", average loss: {epoch_loss:.4f}"
        )
    return model, epoch_loss_values


lr = 1e-2
NUM_EPOCHS = 1000
WANDB_ON = False

if WANDB_ON:
    wandb.init(project="pipeline-test-01", name="denoising-autoencoder-01")

paths_location = "configs/test/LIDC-DRR-test.csv"
paths = pd.read_csv(paths_location, index_col=0).to_numpy()
paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]

model = CustomAutoEncoder(
    spatial_dims=3,
    image_size=64,
    in_channels=1,
    out_channels=1,
    latent_dim=1024,
    channels=(96, 256, 384, 256),
    strides=(2, 2, 2, 2),
)
ds = BaseDataset(
    data=paths, transforms=get_denoising_autoencoder_transforms(size=64, resolution=1.5)
)
data_loader = DataLoader(ds, batch_size=2)
ap, lat, seg = next(iter(data_loader))
out, latent_vec = model(seg['gaus'])
original = seg["orig"]
noisy = seg["gaus"]
printarr(original, noisy, out, latent_vec)

model, epoch_loss_vals = train(model, "gaus", learning_rate=lr, max_epochs=100)
