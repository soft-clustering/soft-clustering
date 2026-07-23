"""Soft Clustering - A library of soft and fuzzy clustering algorithms."""

__version__ = "0.0.3"

from ._afcm import AFCM
from ._afcmadaptive import AFCMAdaptive
from ._afcmSimple import AFCMSimple
from ._bgmm import BGMM
from ._bigclam import BIGCLAM
from ._bnmf import BayesianNMF
from ._cafcm import CAFCM
from ._cafhfcm import CAFHFCM
from ._cdcgs import CDCGS
from ._dmon import DMoN
from ._ecm import ECM
from ._entropyfcm import ENTROPYFCM
from ._fcc import FCC
from ._fcm import FuzzyCMeans as FCM
from ._femifuzzy import FeMIFuzzy
from ._gk import GustafsonKessel as GK
from ._gmm import GaussianMixtureEM as GMM
from ._kfccl import KFCCL
from ._kfcm import KFCM
from ._kmart import KMART
from ._lda import LDA
from ._mbmm import MBMM
from ._mmsb import MMSB
from ._nocd import NOCD
from ._pcm import PossibilisticCMeans as PCM
from ._pfcm import PFCM
from ._plsi import PLSI
from ._rdfkc import RDFKC
from ._rough_k_means import RoughKMeans
from ._rpfkm import RPFKM
from ._scm import SCM
from ._scspa import SCSPA
from ._sfcmep import SFCMEP
from ._shbgf import SHBGF
from ._sisc import SISC
from ._skfcm import SKFCM
from ._smcla import SMCLA
from ._softdbscangm import SoftDBSCANGM
from ._softksc import SoftKSC
from ._wbsc import WBSC

__all__ = [
    "AFCM",
    "AFCMAdaptive",
    "AFCMSimple",
    "BGMM",
    "BIGCLAM",
    "BayesianNMF",
    "CAFCM",
    "CAFHFCM",
    "CDCGS",
    "DMoN",
    "ECM",
    "ENTROPYFCM",
    "FCC",
    "FCM",
    "FeMIFuzzy",
    "GK",
    "GMM",
    "KFCCL",
    "KFCM",
    "KMART",
    "LDA",
    "MBMM",
    "MMSB",
    "NOCD",
    "PCM",
    "PFCM",
    "PLSI",
    "RDFKC",
    "RoughKMeans",
    "RPFKM",
    "SCM",
    "SCSPA",
    "SFCMEP",
    "SHBGF",
    "SISC",
    "SKFCM",
    "SMCLA",
    "SoftDBSCANGM",
    "SoftKSC",
    "WBSC",
]
