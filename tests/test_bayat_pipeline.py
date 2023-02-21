import pandas as pd
from XrayTo3DShape  import TwoDPermuteConcat,BaseDataset,get_nonkasten_transforms
from torch.utils.data import DataLoader
from monai.losses.dice import DiceLoss
import torch
from monai.metrics.meandice import DiceMetric
from monai.transforms import *
import wandb

lr = 1e-2
NUM_EPOCHS = 1000
wandb.init(project='pipeline-test-01',name='bayat-01')

paths_location = 'configs/test/Verse2020-DRR-test.csv'
paths = pd.read_csv(paths_location,index_col=0).to_numpy()
paths = [ {'ap':ap,'lat':lat,'seg':seg} for ap,lat,seg in paths] 

ds = BaseDataset(data=paths, transforms= get_nonkasten_transforms(size=64,resolution=1.5))
data_loader = DataLoader(ds,batch_size=1)
ap,lat,seg = next(iter(data_loader))
ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']


config_bayat = {
    "input_image_size": [64, 64],
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
    "decoder": {
        "in_channels": [64, 64, 64, 64, 64, 32, 16],
        "out_channels": [64, 64, 64, 64, 32, 16, 1],
        "strides": (1, 1, 1, 1, 2, 2, 1),
        "kernel_size": (3,3,3,3,3,3,7),
    },
    "act": "RELU",
    "norm": "BATCH",
    "dropout": 0.0,
    "bias": False
}
model = TwoDPermuteConcat(config_bayat)
pred_tensor = model(ap_tensor,lat_tensor)
print(pred_tensor.shape)
# from torchview import draw_graph

# model_graph = draw_graph(model,input_data=[ap_tensor,lat_tensor],save_graph=True,filename='docs/arch_viz/TwoDPermuteConcat_modelgraph.dot')
loss_function = DiceLoss(sigmoid=True)
optimizer = torch.optim.AdamW(model.parameters(),lr)
dice_metric_evaluator = DiceMetric(include_background=False)

for i in range(NUM_EPOCHS):
    optimizer.zero_grad()
    pred_seg_logits = model(ap_tensor,lat_tensor)

    loss = loss_function(pred_seg_logits,seg_tensor)
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        eval_transform = Compose([
            Activations(sigmoid=True),
            AsDiscrete(threshold=0.5)
        ])
        pred_seg = eval_transform(pred_seg_logits)
        dice_metric_evaluator(y_pred=pred_seg, y=seg_tensor)
    
    acc = dice_metric_evaluator.aggregate()
    dice_metric_evaluator.reset()

    wandb.log({'loss': loss.item(), 'accuracy': acc.item()})
