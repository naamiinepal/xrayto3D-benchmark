# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""custom losses """
from typing import Optional

import torch
from monai.losses.dice import DiceLoss
from torch import Tensor, nn
from torchmetrics.functional import image_gradients


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
    """TODO: debug: does not train well"""
    def __init__(
        self,
        softmax: bool = False,
        sigmoid: bool = False,
        ce_pos_weight: Optional[torch.Tensor] = None,
        lambda_dice: float = 1.0,
        lambda_bce: float = 1.0,
    ) -> None:
        super().__init__()
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce
        self.dice = DiceLoss(
            sigmoid=sigmoid,
            softmax=softmax,
        )
        self.bce = nn.BCEWithLogitsLoss(pos_weight=ce_pos_weight)

    def forward(self, batch_input: torch.Tensor, batch_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD] or B1H[WD].

        Raises:
            ValueError: When number of dimensions for input and target are different.
            ValueError: When number of channels for target is neither 1 nor the same as input.

        """
        if len(batch_input.shape) != len(batch_target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {batch_input.shape} and {batch_target.shape}."
            )

        dice_loss = self.dice(batch_input, batch_target)
        ce_loss = self.bce(batch_input, batch_target)
        total_loss: torch.Tensor = (
            self.lambda_dice * dice_loss + self.lambda_bce * ce_loss
        )

        return total_loss
