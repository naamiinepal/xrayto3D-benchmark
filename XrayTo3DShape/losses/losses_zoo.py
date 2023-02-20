import torch
from torch import nn
from torch import Tensor
from typing import Optional
from torchmetrics.functional import image_gradients
from monai.losses.dice import DiceLoss

class NGCCLoss(nn.Module):
    """
    Normalized Gradient Cross-Correlation (NGCC)
     calculates 1-NGCC
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, net_output: Tensor, gt: Tensor):
        """the input tensors should be 4D torch Tensor #BCHW"""
        net_out_dy, net_out_dx = image_gradients(net_output)
        gt_dy, gt_dx = image_gradients(gt)
        tensor_size = torch.prod(torch.tensor([gt_dy.shape]))
        ngcc = 1 - 0.5 / tensor_size * (
            torch.sum(torch.multiply(gt_dy, net_out_dy))
            + torch.sum(torch.multiply(gt_dx, net_out_dx))
        )

        return ngcc

class DiceCELoss(nn.Module):
    def __init__(self,
    softmax:bool = False,
    sigmoid:bool = False,
    ce_pos_weight:Optional[torch.Tensor]=None,
    lambda_dice:float=1.0,
    lambda_bce:float=1.0,) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.dice = DiceLoss(
            sigmoid=sigmoid,
            softmax=softmax,
        )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=ce_pos_weight)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        dice_loss = self.dice(input, target)
        ce_loss = self.bce(input, target)
        total_loss: torch.Tensor = self.lambda_dice * dice_loss + self.lambda_bce * ce_loss

        return total_loss