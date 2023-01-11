import torch
from torch import nn
from torch import Tensor
from torchmetrics.functional import image_gradients


class NGCCLoss(nn.Module):
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
