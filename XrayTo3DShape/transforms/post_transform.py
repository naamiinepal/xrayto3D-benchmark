# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""data transformation for post-processing model prediction """
from monai.transforms.compose import Compose
from monai.transforms.post.array import Activations
from monai.transforms.post.array import AsDiscrete

post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_transform_onehot = Compose([AsDiscrete(argmax=True)])
