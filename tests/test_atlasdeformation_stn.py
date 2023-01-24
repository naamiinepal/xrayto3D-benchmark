import pandas as pd
from XrayTo3DShape import *
from torch.utils.data import DataLoader
from monai.losses.dice import DiceCELoss
import torch
from monai.metrics.meandice import DiceMetric
from monai.transforms import *
import wandb

lr = 1e-2
NUM_EPOCHS = 1000
WANDB_ON = False

if WANDB_ON:
    wandb.init(project="pipeline-test-01", name="atlasDeformationSTN-01")

atlas_path = "2D-3D-Reconstruction-Datasets/LIDC-test/subjectwise/LIDC-0001/derivatives/seg_roi/LIDC-0001_vert-15-seg-vert_msk.nii.gz"
paths_location = "configs/test/LIDC-DRR-test.csv"
paths = pd.read_csv(paths_location, index_col=0).to_numpy()
paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]

ds = AtlasDeformationDataset(
    data=paths,
    transforms=get_atlas_deformation_transforms(size=64, resolution=1.5),
    atlas_path=atlas_path,
)
data_loader = DataLoader(ds, batch_size=1)
ap, lat, seg, atlas = next(iter(data_loader))
ap_tensor, lat_tensor, seg_tensor, atlas_tensor = ap["ap"], lat["lat"], seg["seg"], atlas['atlas']


config = {        
    "encoder": {
        "in_channels": [1, 16, 32, 32, 32, 32],
        "out_channels": [16, 32, 32, 32, 32, 32],
        "strides": [2, 2, 1, 1, 1, 1],
        "kernel_size": 7,
    },
    "ap_expansion": {
        "in_channels": [32, 32, 32, 32],
        "out_channels": [32, 32, 32, 32],
        "strides": ((2, 1, 1),) * 4,
        "kernel_size": 3,
    },
    "lat_expansion": {
        "in_channels": [32, 32, 32, 32],
        "out_channels": [32, 32, 32, 32],
        "strides": ((1, 1, 2),) * 4,
        "kernel_size": 3,
    },
    'affine': {
        'in_channels':[16384,4096,1024],
        'out_channels': [4096,1024,32,]
    },
    'kernel_size':5,
    'act': 'RELU',
    'norm': 'BATCH',
    'dropout' : 0.0,
}
model = AtlasDeformationSTN(config)
pred_tensor = model(ap_tensor, lat_tensor,atlas_tensor)
print(pred_tensor.shape)

loss_function = DiceCELoss(sigmoid=True)
ngcc_loss_function = NGCCLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr)
dice_metric_evaluator = DiceMetric(include_background=False)

for i in range(NUM_EPOCHS):
    optimizer.zero_grad()
    pred_seg_logits = model(ap_tensor, lat_tensor,atlas_tensor)

    loss = loss_function(pred_seg_logits, seg_tensor)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        eval_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
        ap_pred, _, lat_pred = get_projectionslices_from_3d(pred_seg_logits)
        ngcc_loss = ngcc_loss_function(ap_pred, ap_tensor)
        pred_seg = eval_transform(pred_seg_logits)
        dice_metric_evaluator(y_pred=pred_seg, y=seg_tensor)

    acc = dice_metric_evaluator.aggregate()
    dice_metric_evaluator.reset()
    if WANDB_ON:
        wandb.log({"loss": loss.item(), "accuracy": acc.item(), "1-NGCC": ngcc_loss.item()})
    print(f"loss {loss.item():.4f} accuracy {acc.item():.4f} NGCC {ngcc_loss.item():.4f}")
