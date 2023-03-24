import pandas as pd
import torch
from monai.losses.dice import DiceLoss
from monai.metrics.meandice import DiceMetric
from monai.transforms.compose import Compose
from monai.transforms.post.array import Activations, AsDiscrete
from torch.utils.data import DataLoader

import wandb
from XrayTo3DShape import (
    get_nonkasten_transforms,
    BaseDataset,
    MultiScale2DPermuteConcat,
    NGCCLoss,
    get_projectionslices_from_3d,
    get_model_config,
    printarr,
)

lr = 1e-2
NUM_EPOCHS = 1000
WANDB_ON = False
BATCH_SIZE = 2
if WANDB_ON:
    wandb.init(project="pipeline-test-01", name="TwoDPermuteConcatMultiScale-NGCC-01")

paths_location = "configs/test/Verse2020-DRR-test.csv"
paths = pd.read_csv(paths_location, index_col=0).to_numpy()
paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]

ds = BaseDataset(
    data=paths, transforms=get_nonkasten_transforms(size=64, resolution=1.5)
)
data_loader = DataLoader(ds, batch_size=BATCH_SIZE)
ap, lat, seg = next(iter(data_loader))
ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]


config = get_model_config(MultiScale2DPermuteConcat.__name__, 128, False)


x_ray_img = torch.zeros((1, 1, 128, 128))

model = MultiScale2DPermuteConcat(config)
out = model(x_ray_img, x_ray_img)
printarr(out, ap_tensor, lat_tensor, seg_tensor)

loss_function = DiceLoss(sigmoid=True)
ngcc_loss_function = NGCCLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr)
dice_metric_evaluator = DiceMetric(include_background=False)

for i in range(NUM_EPOCHS):
    optimizer.zero_grad()
    pred_seg_logits = model(ap_tensor, lat_tensor)
    print(f"output {pred_seg_logits.shape}")
    loss = loss_function(pred_seg_logits, seg_tensor)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        eval_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        ap_pred, _, lat_pred = get_projectionslices_from_3d(pred_seg_logits)
        # ngcc_loss = ngcc_loss_function(ap_pred,ap_tensor)
        pred_seg = eval_transform(pred_seg_logits)
        dice_metric_evaluator(y_pred=pred_seg, y=seg_tensor)

    acc = dice_metric_evaluator.aggregate()
    dice_metric_evaluator.reset()
    if WANDB_ON:
        wandb.log(
            {
                "loss": loss.item(),
                "accuracy": acc.item(),
            }
        )
    print(f"loss {loss.item():.4f} accuracy {acc.item():.4f}")
