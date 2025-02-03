# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""Base class with common functionality for encoder-decoder based methods"""
from typing import Any, Optional, Tuple

import pytorch_lightning as pl
import torch
from monai.metrics.meandice import compute_dice

import wandb
from XrayTo3DShape import post_transform

from ..utils import reproject, to_numpy


class BaseExperiment(pl.LightningModule):
    """
    Base class for training Encoder-Decoder architecture
    override ``get_input_out_from_batch`` as required
    training regime for many of these architecture is same except how AP/LAT images are processed
    """

    def __init__(
        self, model, optimizer=None, loss_function=None, batch_size=None, **kwargs: Any
    ) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.loss_function = loss_function
        self.batch_size = batch_size

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        """subclasses should override this"""
        raise NotImplementedError()

    def get_segmentation_meta_dict(self, batch):
        """util for extracting meta data from batch"""
        ap, lat, seg = batch
        return seg["seg_meta_dict"]

    def training_step(self, batch, batch_idx):
        """single step backprop"""
        batch_input, output = self.get_input_output_from_batch(batch)
        pred_logits = self.model(*batch_input)
        loss = self.loss_function(pred_logits, output)
        with torch.no_grad():
            pred = post_transform(pred_logits.detach())
            dice_metric = torch.mean(compute_dice(pred, output))

        self.log(
            "train/loss",
            loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "train/dice",
            dice_metric.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        batch_input, output = self.get_input_output_from_batch(batch)
        pred_logits = self.model(*batch_input)
        loss = self.loss_function(pred_logits, output)
        pred = post_transform(pred_logits)
        if self.global_step % 20 and batch_idx == 0:
            self.log_3d_images(pred, label="val/predictions")  # type: ignore
            self.log_3d_images(output, label="val/groundtruth")  # type: ignore

        dice_metric = torch.mean(compute_dice(pred, output, ignore_empty=False))
        self.log(
            "val/loss",
            loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log(
            "val/dice",
            dice_metric.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        batch_input, output = self.get_input_output_from_batch(batch)
        pred_logits = self.model(*batch_input)
        pred = post_transform(pred_logits)

        out = {}
        seg_meta_dict = self.get_segmentation_meta_dict(batch)

        out["pred"] = pred
        out["gt"] = output
        out["seg_meta_dict"] = seg_meta_dict

        return out

    def configure_optimizers(self):
        return self.optimizer

    def log_3d_images(self, predictions, label):
        """log projections of 3d volume into wandb"""
        projections = [reproject(volume.squeeze(), 0) for volume in predictions]
        wandb.log(
            {
                f"{label}": [
                    wandb.Image(to_numpy(image.float())) for image in projections
                ]
            }
        )
