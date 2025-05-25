"""Configure global settings and get information about the working environment."""

__version__ = "0.0.1"

from ._nocd import NOCD
from ._wbsc import WBSC
__all__ = ["NOCD", "WBSC"]

