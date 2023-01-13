import pandas as pd
from XrayTo3DShape  import *
from torch.utils.data import DataLoader
from monai.losses.dice import DiceCELoss
import torch
from monai.metrics.meandice import DiceMetric
from monai.transforms import *
import wandb

lr = 1e-2
NUM_EPOCHS = 1000
WANDB_ON = True

if WANDB_ON:
    wandb.init(project='pipeline-test-01',name='TwoDPermuteConcatMultiScale-NGCC-01')

paths_location = 'configs/test/LIDC-DRR-test.csv'
paths = pd.read_csv(paths_location,index_col=0).to_numpy()
paths = [ {'ap':ap,'lat':lat,'seg':seg} for ap,lat,seg in paths] 

ds = BaseDataset(data=paths, transforms= get_nonkasten_transforms(size=64,resolution=1.5))
data_loader = DataLoader(ds,batch_size=1)
ap,lat,seg = next(iter(data_loader))
ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']


config = {
    "in_shape": (1,64,64),
    "kernel_size":3,
    "act":'RELU',
    "norm":"BATCH",
    "encoder":{
        "kernel_size":(3,)*4,
        "strides":(1,2,2,2),   # keep the first element of the strides 1 so the input and output shape match

    },
    "decoder": {
        "out_channel":16,
        "kernel_size":3
    }
}



x_ray_img = torch.zeros(1, *config['in_shape'])

model = TwoDPermuteConcatMultiScale(config)
out = model(x_ray_img,x_ray_img)
print(out.shape)

loss_function = DiceCELoss(sigmoid=True)
ngcc_loss_function = NGCCLoss()
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
        ap_pred, _, lat_pred = get_projectionslices_from_3d(pred_seg_logits)
        ngcc_loss = ngcc_loss_function(ap_pred,ap_tensor)
        pred_seg = eval_transform(pred_seg_logits)
        dice_metric_evaluator(y_pred=pred_seg, y=seg_tensor)
    
    acc = dice_metric_evaluator.aggregate()
    dice_metric_evaluator.reset()
    if WANDB_ON:
        wandb.log({'loss': loss.item(), 'accuracy': acc.item(),'1-NGCC': ngcc_loss.item()})
    print(f'loss {loss.item():.4f} accuracy {acc.item():.4f} NGCC {ngcc_loss.item():.4f}')
