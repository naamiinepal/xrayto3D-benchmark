import pandas as pd
import torch
from monai.losses.dice import DiceCELoss
from monai.metrics.meandice import DiceMetric
from monai.networks.nets.unet import UNet
from monai.transforms.compose import Compose
from monai.transforms.post.array import Activations, AsDiscrete
from torch.utils.data import DataLoader

import wandb
from XrayTo3DShape import BaseDataset, get_kasten_transforms

lr = 1e-2
NUM_EPOCHS = 1000
WANDB_ON = False

if WANDB_ON:
    wandb.init(project="pipeline-test-01", name="kasten-01-res-units-0")

paths_location = "configs/test/LIDC-DRR-test.csv"
paths = pd.read_csv(paths_location, index_col=0).to_numpy()
paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]

ds = BaseDataset(data=paths, transforms=get_kasten_transforms(size=64, resolution=1.5))
data_loader = DataLoader(ds, batch_size=1)
ap, lat, seg = next(iter(data_loader))
ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]


config_kasten = {
    "in_channels": 2,
    "out_channels": 1,
    "channels": (8, 16, 32),
    "strides": (2,2,2),
    "act": "RELU",
    "norm": "BATCH",
    "num_res_units": 0,
}
model = UNet(spatial_dims=3, **config_kasten)
input_volume = torch.cat((ap_tensor,lat_tensor),1)
pred_tensor = model(input_volume)
print(pred_tensor.shape)

loss_function = DiceCELoss(sigmoid=True)
optimizer = torch.optim.AdamW(model.parameters(), lr)
dice_metric_evaluator = DiceMetric(include_background=False)

for i in range(NUM_EPOCHS):
    optimizer.zero_grad()
    input_volume = torch.cat((ap_tensor,lat_tensor),1)
    pred_seg_logits = model(input_volume)

    loss = loss_function(pred_seg_logits, seg_tensor)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        eval_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        pred_seg = eval_transform(pred_seg_logits)
        dice_metric_evaluator(y_pred=pred_seg, y=seg_tensor)

    acc = dice_metric_evaluator.aggregate()
    dice_metric_evaluator.reset()

    if WANDB_ON:
        wandb.log(
            {"loss": loss.item(), "accuracy": acc.item()}
        )
    print(
        f"loss {loss.item():.4f} accuracy {acc.item():.4f}"
    )
