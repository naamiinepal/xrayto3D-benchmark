"""Architecture specific training regime

VolumeAsInput
https://arxiv.org/abs/2004.00871
End-to-end convolutional neural network for 3D reconstruction of knee bones from bi-planar X-ray images

ParallelHead
https://arxiv.org/pdf/2007.06612
Inferring the 3D standing spine posture from 2D radiographs

TLPredictor
https://arxiv.org/abs/1603.08637
Learning a Predictable and Generative Vector Representation for Objects
"""
from typing import Any, Optional, Tuple

import torch
from monai.metrics.meandice import compute_dice

import wandb

from ..architectures import CustomAutoEncoder
from ..transforms import post_transform
from ..utils import reproject, to_numpy
from .base_experiment import BaseExperiment
from ..losses import l1_loss


class VolumeAsInputExperiment(BaseExperiment):
    """Merge AP and LAT volume as 2-channel 3D volume"""

    def __init__(
        self, model, optimizer=None, loss_function=None, batch_size=None, **kwargs: Any
    ) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]
        input = torch.cat((ap_tensor, lat_tensor), 1)
        return [input], seg_tensor


class ParallelHeadsExperiment(BaseExperiment):
    """keep AP and LAT 2D images separate"""

    def __init__(
        self, model, optimizer=None, loss_function=None, batch_size=None, **kwargs: Any
    ) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]
        batch_input = (ap_tensor, lat_tensor)
        return batch_input, seg_tensor


class AutoencoderExperiment(BaseExperiment):
    """train a autoencoder with 1D bottlneck"""

    def __init__(
        self,
        model,
        optimizer=None,
        loss_function=None,
        batch_size=None,
        make_sparse=False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)
        self.SPARSITY_REG_STRENGTH = 1e-3
        self.make_sparse = make_sparse

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        _, _, seg = batch
        noisy_seg, clean_seg = seg["gaus"], seg["orig"]
        return [
            noisy_seg
        ], clean_seg  # the input has to be put inside a list as a hack to keep the interface same for all type of models self.model(*input)

    def training_step(self, batch, batch_idx):
        batch_input, output = self.get_input_output_from_batch(batch)
        pred_logits, latent_vector = self.model(*batch_input)
        supervised_loss = self.loss_function(pred_logits, output)
        if self.make_sparse:
            sparsity_loss = l1_loss(latent_vector)
            total_loss = supervised_loss + self.SPARSITY_REG_STRENGTH * sparsity_loss
        else:
            total_loss = supervised_loss
        self.log(
            "train/loss",
            total_loss.item(),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        if self.global_step % 20 and batch_idx == 0:
            with torch.no_grad():
                self.log_3d_images(pred_logits.detach(), label="train/predictions")
                self.log_3d_images(output, label="train/groundtruth")
        return total_loss

    def validation_step(self, batch, batch_idx):
        batch_input, output = self.get_input_output_from_batch(batch)
        pred_logits, latent_vector = self.model(*batch_input)
        supervised_loss = self.loss_function(pred_logits, output)
        if self.make_sparse:
            sparsity_loss = l1_loss(latent_vector)
            total_loss = supervised_loss + self.SPARSITY_REG_STRENGTH * sparsity_loss
        else:
            total_loss = supervised_loss
        self.log(
            "val/loss",
            total_loss.item(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        if self.global_step % 20 and batch_idx == 0:
            self.log_3d_images(pred_logits.detach(), label="val/predictions")
            self.log_3d_images(output, label="val/groundtruth")

    def log_3d_images(self, predictions, label):
        projections = [reproject(volume.squeeze(), 0) for volume in predictions]
        wandb.log(
            {
                f"{label}": [
                    wandb.Image(to_numpy(image.float())) for image in projections
                ]
            }
        )


class TLPredictorExperiment(BaseExperiment):
    """train the predictor in the TL-Embedding network
    https://arxiv.org/abs/1603.08637
    Learning a Predictable and Generative Vector Representation for Objects, Girdhar et.al
    """

    def __init__(
        self, model, optimizer=None, loss_function=None, batch_size=None, **kwargs: Any
    ) -> None:
        super().__init__(model, optimizer, loss_function, batch_size, **kwargs)
        self.TNet: CustomAutoEncoder

    def set_decoder(self, autoencoder_model: CustomAutoEncoder):
        self.TNet = autoencoder_model
        self.TNet.eval()  # set to eval mode

    def get_input_output_from_batch(self, batch) -> Tuple[Any, torch.Tensor]:
        ap, lat, seg = batch
        ap_tensor, lat_tensor, seg_tensor = ap["ap"], lat["lat"], seg["seg"]
        batch_input = torch.cat((ap_tensor, lat_tensor), 1)
        return [batch_input], seg_tensor

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Any:
        batch_input, output = self.get_input_output_from_batch(batch)
        pred_latent_vec = self.model(*batch_input)
        pred_logits = self.TNet.latent_vec_decode(pred_latent_vec)
        pred = post_transform(pred_logits)

        out = {}
        seg_meta_dict = self.get_segmentation_meta_dict(batch)

        out["pred"] = pred
        out["gt"] = output
        out["seg_meta_dict"] = seg_meta_dict

        return out

    def training_step(self, batch, batch_idx):
        batch_input, output = self.get_input_output_from_batch(batch)
        pred_latent_vec = self.model(*batch_input)

        pred_logits = self.TNet.latent_vec_decode(pred_latent_vec)
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
        batch_input, batch_output = self.get_input_output_from_batch(batch)
        pred_latent_vec = self.model(*batch_input)
        pred_logits = self.TNet.latent_vec_decode(pred_latent_vec)
        loss = self.loss_function(pred_logits, batch_output)
        pred = post_transform(pred_logits.detach())
        dice_metric = torch.mean(compute_dice(pred, batch_output, ignore_empty=False))
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
