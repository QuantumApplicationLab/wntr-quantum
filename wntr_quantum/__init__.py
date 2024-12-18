"""Documentation about wntr_quantum."""
import logging
from . import sim

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Nicolas Renaud"
__email__ = "n.renaud@esciencecenter.nl"
__version__ = "0.1.0"

__all__ = [
    "sim",
]
