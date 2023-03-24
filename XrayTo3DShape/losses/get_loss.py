"""interface for losses zoo"""
import torch
from monai.losses.dice import DiceLoss
from torch.nn import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from .hausdorff import HausdorffDTLoss, HausdorffERLoss
from .losses_zoo import DiceCELoss

pos_weights_dict = {"hip": 719, "femur": 612, "vertebra": 23, "ribs": 5231, "rib": 5231}


def get_loss(loss_name, **kwargs):
    """given name of loss and passed arguments,
    return a callable loss function"""
    if loss_name == MSELoss.__name__:
        return MSELoss()
    if loss_name == BCEWithLogitsLoss.__name__:
        return get_WCE(kwargs["anatomy"], kwargs["image_size"])
    if loss_name == BCELoss.__name__:
        return BCELoss()
    elif loss_name == CrossEntropyLoss.__name__:
        return get_CE(kwargs["anatomy"], kwargs["image_size"])
    elif loss_name == DiceLoss.__name__:
        return DiceLoss(sigmoid=True)
    elif loss_name == DiceCELoss.__name__:
        return get_DiceCE(**kwargs, sigmoid=True)
    elif loss_name == HausdorffDTLoss.__name__:
        return HausdorffDTLoss(device=kwargs["device"])
    elif loss_name == HausdorffERLoss.__name__:
        return HausdorffERLoss(device=kwargs["device"])  # broken
    else:
        raise ValueError(f"invalid loss name {loss_name}")


def get_WCE(anatomy, image_size):
    """Weighted cross-entropy loss"""
    pos_weight = torch.full(
        [1, image_size, image_size, image_size], pos_weights_dict[anatomy]
    )
    return BCEWithLogitsLoss(pos_weight=pos_weight)


def get_CE(anatomy, image_size):
    """Regular cross-entropy loss"""
    # pos_weight = torch.full([1,image_size,image_size,image_size],pos_weights_dict[anatomy])
    # return CrossEntropyLoss(weight=pos_weight)
    return CrossEntropyLoss()


def get_DiceCE(
    anatomy, image_size, sigmoid=True, softmax=False, lambda_dice=1.0, lambda_bce=1.0
):
    """Dice+CE Loss"""
    pos_weight = torch.full(
        [1, image_size, image_size, image_size], pos_weights_dict[anatomy]
    )
    return DiceCELoss(
        softmax=softmax,
        sigmoid=sigmoid,
        ce_pos_weight=pos_weight,
        lambda_dice=lambda_dice,
        lambda_bce=lambda_bce,
    )
