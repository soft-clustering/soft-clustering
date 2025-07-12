"""Configure global settings and get information about the working environment."""

__version__ = "0.0.1"

from ._nocd import NOCD
from ._wbsc import WBSC
from ._pfcm import PFCM
from ._rough_k_means import RoughKMeans
from ._lda import LDA
from ._plsi import PLSI
from ._rdfkc import RDFKC
from ._kfcm import KFCM
__all__ = ["NOCD", "WBSC", "PFCM", "RoughKMeans", "KFCM"]

__all__ = ["NOCD", "WBSC", "PFCM", "RoughKMeans", "RDFKC"]