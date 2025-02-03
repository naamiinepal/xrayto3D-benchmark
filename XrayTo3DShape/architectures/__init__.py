# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.

 
from ..utils.registry import ARCHITECTURES
name2arch = ARCHITECTURES

from .atlas_deformation_stn import AtlasDeformationSTN
from .autoencoder import AutoEncoder1DEmbed, Encoder1DEmbed
from .autoencoder_v2 import CustomAutoEncoder, TLPredictor
from .get_model import get_model, get_model_config
from .onedconcat import OneDConcat, OneDConcatModel
from .twodpermuteconcat import TwoDPermuteConcat, TwoDPermuteConcatModel
from .twodpermuteconcatmultiscale import MultiScale2DPermuteConcat
from .arch_utils import calculate_1d_vec_channels