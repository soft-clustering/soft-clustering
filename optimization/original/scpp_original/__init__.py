"""Reference (pre-optimization) SCPP implementations.

Verbatim copies of the algorithm modules as they stood before the optimization
study, together with the unmodified ``_base.py`` protocol they rely on. This
package exists so that optimized implementations can be verified and
benchmarked against a stable baseline.

It is deliberately **outside** the production import path: nothing in
``soft_clustering/`` imports from here, and this package is not distributed.

Import it exactly as you would the real package::

    import scpp_original
    model = scpp_original.SoftDBSCANGM()
"""

from ._base import BaseSoftClusterer  # noqa: F401

_ESTIMATORS = {
    "KFCCL": "._kfccl",
    "KFCM": "._kfcm",
    "KMART": "._kmart",
    "MBMM": "._mbmm",
    "SoftDBSCANGM": "._softdbscangm",
}
_ALIASES = {}


def __getattr__(name):
    module_path = _ESTIMATORS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_path, __name__)
    obj = getattr(module, _ALIASES.get(name, name))
    globals()[name] = obj
    return obj


def __dir__():
    return sorted(list(globals()) + list(_ESTIMATORS))


__all__ = sorted(_ESTIMATORS)
