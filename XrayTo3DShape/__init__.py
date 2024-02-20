from .utils.registry import import_all_modules_for_register

import_all_modules_for_register()

from .utils import *
from .datasets import *
from .architectures import *
from .losses import *
from .transforms import *
from .experiments import *
from .consts import *
