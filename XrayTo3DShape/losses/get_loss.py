from torch.nn import BCEWithLogitsLoss
from monai.losses.dice import DiceLoss,DiceCELoss
import torch

def get_loss(loss_name,**kwargs):
    if loss_name == BCEWithLogitsLoss.__name__:
        return get_WCE(**kwargs)
    elif loss_name == DiceLoss.__name__:
        return DiceLoss(sigmoid=True)
    elif loss_name == DiceCELoss.__name__:
        return DiceCELoss()
    else:
        raise ValueError(f'invalid loss name {loss_name}')

def get_WCE(anatomy,image_size):
    # Weighted cross-entropy loss
    pos_weights_dict = {'hip':719,'femur':612,'vertebra':23,'rib':5231}
    pos_weight = torch.full([1,image_size,image_size,image_size],pos_weights_dict[anatomy])
    return BCEWithLogitsLoss(pos_weight=pos_weight)
