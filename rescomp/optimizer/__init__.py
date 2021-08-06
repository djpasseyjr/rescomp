"""
An interface for easily performing hyperparameter optimization on a reservoir computer using the sherpa package.
The main object of note is the ResCompOptimizer class, which implements this.

Also of note is the System class, which can be used for defining systems.
"""

from .optimizer_controller import ResCompOptimizer
from .optimizer_functions import *
from .optimizer_systems import get_res_defaults, load_from_file, loadprior, get_system
from .templates import random_slice, System