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
from ._kfccl import KFCCL
from ._scm import SCM
from ._fcm import FuzzyCMeans as FCM
from ._gmm import GaussianMixtureEM as GMM
from ._pcm import PossibilisticCMeans as PCM
from ._gk import GustafsonKessel as GK


__all__ = [
    "NOCD",
    "WBSC",
    "PFCM",
    "RoughKMeans",
    "LDA",
    "PLSI",
    "RDFKC",
    "KFCM",
    "KFCCL",
    "SCM",
    "FCM",
    "GMM",
    "PCM",
    "GK"
]
