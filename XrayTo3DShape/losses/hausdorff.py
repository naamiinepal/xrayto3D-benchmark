"""
from https://github.com/JunMa11/SegLoss/blob/master/losses_pytorch/hausdorff.py

Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf

copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss
"""
import cv2 as cv
import numpy as np
import torch
from scipy.ndimage import convolve
from scipy.ndimage.morphology import distance_transform_edt as edt
from torch import nn


class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, device, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha
        self.device = device

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in img:
            fg_mask = batch > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist: np.ndarray = edt(fg_mask)  # type: ignore
                bg_dist: np.ndarray = edt(bg_mask)  # type: ignore

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        pred_dt = (
            torch.from_numpy(self.distance_field(pred.detach().cpu().numpy()))
            .float()
            .to(device=self.device)
        )
        target_dt = (
            torch.from_numpy(self.distance_field(target.detach().cpu().numpy()))
            .float()
            .to(device=self.device)
        )

        pred_error = (pred - target) ** 2
        distance = pred_dt**self.alpha + target_dt**self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, device, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.device = device
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroded = np.zeros_like(bound)
        erosions = []

        for batch, _ in enumerate(bound):
            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):
                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroded[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroded, erosions
        else:
            return eroded

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroded, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroded.mean(), erosions

        else:
            eroded = (
                torch.from_numpy(
                    self.perform_erosion(
                        pred.detach().cpu().numpy(),
                        target.detach().cpu().numpy(),
                        debug,
                    )
                )
                .float()
                .to(device=self.device)
            )

            loss = eroded.mean()

            return loss
