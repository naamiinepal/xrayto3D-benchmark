import torch
from XrayTo3DShape import get_kasten_transforms, BaseDataset, Encoder1DEmbed
from torch.utils.data import DataLoader
import pandas as pd
import wandb
from monai.transforms.compose import Compose
from monai.transforms.post.array import AsDiscrete
from monai.metrics.meandice import DiceMetric
from monai.losses.dice import DiceCELoss

denoising_ae_path = "tests/denoising_ae_epoch100.pth"
ae_model = torch.load(denoising_ae_path)

paths_location = "configs/test/LIDC-DRR-test.csv"
paths = pd.read_csv(paths_location, index_col=0).to_numpy()
paths = [{"ap": ap, "lat": lat, "seg": seg} for ap, lat, seg in paths]

ds = BaseDataset(data=paths, transforms=get_kasten_transforms(size=64, resolution=1.5))
data_loader = DataLoader(ds, batch_size=1)
ap, lat, seg = next(iter(data_loader))
print(ap["ap"].shape, lat["lat"].shape, seg["seg"].shape)

with torch.no_grad():
    low_dim_embedding_vec = ae_model.encode_forward(seg["seg"])
    seg_out = ae_model.decode_forward(low_dim_embedding_vec).detach().numpy()
    print(low_dim_embedding_vec.shape, seg_out.shape)

encoder_model = Encoder1DEmbed(
    spatial_dims=3,
    in_shape=(2, 64, 64, 64),
    out_channels=1,
    latent_size=1024,
    channels=(32, 64, 128, 256),
    strides=(2, 2, 2, 2),
)

lr = 1e-2
NUM_EPOCHS = 500
WANDB_ON = False
if WANDB_ON:
    wandb.init(project="pipeline-test-01", name="tl-predictor-01")

loss_function = DiceCELoss(sigmoid=False)
optimizer = torch.optim.AdamW(encoder_model.parameters(), lr=lr)
dice_metric_evaluator = DiceMetric()

for i in range(NUM_EPOCHS):
    optimizer.zero_grad()
    ap_1d_vec = encoder_model(torch.cat((ap["ap"], lat["lat"]), dim=1))
    # print(ap_1d_vec)
    # print(low_dim_embedding_vec)
    pred_seg_activations = ae_model.decode_forward(ap_1d_vec)
    loss = loss_function(pred_seg_activations, seg["seg"])
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        eval_transform = Compose([AsDiscrete(threshold=0.5)])
        pred_seg_activations = ae_model.decode_forward(ap_1d_vec)
        pred_seg = eval_transform(pred_seg_activations)
        dice_metric_evaluator(y_pred=pred_seg, y=seg["seg"])
    acc = dice_metric_evaluator.aggregate()
    dice_metric_evaluator.reset()

    print(f"loss {loss.item():.4f} acc: {acc.item():.4f}")
    if WANDB_ON:
        wandb.log({"loss": loss.item(), "accuracy": acc.item()})
