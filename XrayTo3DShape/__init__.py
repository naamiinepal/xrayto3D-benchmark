# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


from .utils.registry import import_all_modules_for_register

import_all_modules_for_register()

from .utils import *
from .datasets import *
from .architectures import *
from .losses import *
from .transforms import *
from .experiments import *
from .consts import *
