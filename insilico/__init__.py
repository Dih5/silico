"""insilico - Python package to handle in silico experiments"""

__version__ = '0.1.0'
__author__ = 'Dih5 <dihedralfive@gmail.com>'

from .base import Experiment, Variable, SubExperiment
from .plot import highlight_max, highlight_threshold
from .analysis import paired_t_test