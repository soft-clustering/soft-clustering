# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

# Configuration file for the Sphinx documentation builder.

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version
from pathlib import Path

# Import the benchmarking modules before autodoc mocks torch: scikit-learn's
# array-API layer inspects ``torch`` when it is present in ``sys.modules`` and
# fails on the mock, which would make these modules unimportable mid-build.
import soft_clustering.benchmarking.datasets  # noqa: E402,F401
import soft_clustering.benchmarking.metrics  # noqa: E402,F401

# -- Project information

project = "SCPP — Soft Clustering Python Package"
copyright = "2025-2026, the SCPP developers"
author = "The SCPP developers"

# Read from the installed distribution rather than restating it here, so the
# documentation cannot claim a version the package does not have.
try:
    release = _package_version("soft-clustering")
except PackageNotFoundError:  # building against a source tree, not an install
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.duration",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_parser",
]

# The four deep estimators import torch at module level. Mocking it lets their
# docstrings be rendered without installing the deep extra on Read the Docs.
autodoc_mock_imports = ["torch"]
autodoc_member_order = "bysource"
# Several estimators document their parameters on __init__.
autoclass_content = "both"
autodoc_typehints = "description"
autodoc_default_options = {"members": False, "undoc-members": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = True  # a few older modules use Args: blocks
napoleon_use_rtype = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- Options for the linkcheck builder

# Hosts whose failures say nothing about this project. Checking them produces
# noise that hides the failures actually worth acting on.
linkcheck_ignore = [
    # Self-references to this repository; GitHub rate-limits shared CI runners.
    r"https://github\.com/.*",
    # Academic publishers serve 403 to automated clients. DOIs are included
    # because they redirect to exactly these hosts.
    r"https://doi\.org/.*",
    r"https://dl\.acm\.org/.*",
    r"https://ieeexplore\.ieee\.org/.*",
    r"https://link\.springer\.com/.*",
    r"https://www\.sciencedirect\.com/.*",
    r"https://onlinelibrary\.wiley\.com/.*",
]
linkcheck_timeout = 15
linkcheck_retries = 2
linkcheck_rate_limit_timeout = 30.0
linkcheck_report_timeouts_as_broken = False

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"
epub_show_urls = "footnote"


# -- Generated algorithm pages ------------------------------------------------
#
# One page per estimator, generated from soft_clustering._registry and the
# estimator's own docstring at every build, so the documentation cannot drift
# from the code. The generated files are not committed (see .gitignore).

_ALGORITHM_DIR = Path(__file__).parent / "algorithms"

# Public names whose class has a longer name; documented under the class name
# so that autodoc does not render them as "alias of ...".
_CLASS_NAMES = {
    "FCM": "FuzzyCMeans",
    "GK": "GustafsonKessel",
    "GMM": "GaussianMixtureEM",
    "PCM": "PossibilisticCMeans",
}


def _module_docstring_needed(class_name: str) -> str | None:
    """The estimator's module, if its class docstring lacks the references.

    Decided from the source with ``ast`` so that building the documentation
    does not require the optional deep-learning dependencies.
    """
    import ast

    import soft_clustering

    module = soft_clustering._ESTIMATORS[class_name]
    path = Path(soft_clustering.__file__).parent / f"{module.lstrip('.')}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    target = _CLASS_NAMES.get(class_name, class_name)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == target:
            doc = ast.get_docstring(node) or ""
            if "References" not in doc and ast.get_docstring(tree):
                return f"soft_clustering{module}"
    return None


def _rst_escape(text: str) -> str:
    return text.replace("*", r"\*").replace("|", r"\|")


def _generate_algorithm_pages(app) -> None:
    from soft_clustering import DEEP_ESTIMATORS
    from soft_clustering._registry import (
        ALGORITHMS,
        CLUSTER_COUNT,
        OUTPUT_KINDS,
        STATUS,
    )

    _ALGORITHM_DIR.mkdir(exist_ok=True)
    names = sorted(ALGORITHMS, key=str.lower)
    for name in names:
        a = ALGORITHMS[name]
        target = _CLASS_NAMES.get(name, name)
        reference = f"`{a.reference} <{a.link}>`__" if a.link else a.reference
        alias = (
            f"\nImport as ``from soft_clustering import {name}`` (the class is "
            f"``{target}``).\n"
            if name in _CLASS_NAMES
            else f"\nImport as ``from soft_clustering import {name}``.\n"
        )
        rows = [
            ("Family", a.family),
            ("Input", a.input),
            ("Output", f"{a.output} — {OUTPUT_KINDS[a.output]}"),
            ("Number of clusters", CLUSTER_COUNT[a.n_clusters]),
            ("Verification status", f"**{a.status}** — {STATUS[a.status]}"),
            ("Reference", reference),
        ]
        if a.note:
            rows.append(("Note", a.note))
        if name in DEEP_ESTIMATORS:
            rows.append(
                (
                    "Requires",
                    'the ``deep`` extra: ``pip install "soft-clustering[deep]"``',
                )
            )
        table = "\n".join(
            f"   * - {key}\n     - {_rst_escape(value) if key != 'Reference' else value}"
            for key, value in rows
        )
        title = f"{name} — {a.method}"
        module = _module_docstring_needed(name)
        background = f".. automodule:: {module}\n   :no-members:\n\n" if module else ""
        page = (
            f".. This page is generated by docs/source/conf.py; do not edit.\n\n"
            f"{title}\n{'=' * len(title)}\n{alias}\n"
            f".. list-table::\n   :widths: 25 75\n\n{table}\n\n"
            f"{background}"
            f".. autoclass:: soft_clustering.{target}\n"
            f"   :members: fit, fit_predict, predict, predict_proba\n"
        )
        path = _ALGORITHM_DIR / f"{name.lower()}.rst"
        if not path.exists() or path.read_text(encoding="utf-8") != page:
            path.write_text(page, encoding="utf-8")

    def index_row(name):
        a = ALGORITHMS[name]
        deep = " ⚡" if name in DEEP_ESTIMATORS else ""
        return (
            f"   * - :doc:`{name}{deep} <{name.lower()}>`\n"
            f"     - {_rst_escape(a.method)}\n"
            f"     - {_rst_escape(a.input)}\n"
            f"     - {a.output}\n"
            f"     - {a.n_clusters}\n"
            f"     - {a.status}"
        )

    index = (
        ".. This page is generated by docs/source/conf.py; do not edit.\n\n"
        "Algorithms\n==========\n\n"
        "Every estimator SCPP provides, generated from "
        "``soft_clustering/_registry.py``. ⚡ marks estimators that need the "
        "``deep`` extra. The **Output** codes say what ``memberships_`` holds:\n\n"
        + "\n".join(f"- **{k}** — {v}" for k, v in OUTPUT_KINDS.items())
        + "\n\nThe **Status** levels say how far each implementation has been "
        "verified:\n\n"
        + "\n".join(f"- **{k}** — {v}." for k, v in STATUS.items())
        + "\n\n.. list-table::\n   :header-rows: 1\n   :widths: 16 40 22 7 10 10\n\n"
        "   * - Estimator\n     - Method\n     - Input\n     - Output\n"
        "     - K\n     - Status\n"
        + "\n".join(index_row(n) for n in names)
        + "\n\n.. toctree::\n   :hidden:\n\n"
        + "\n".join(f"   {n.lower()}" for n in names)
        + "\n"
    )
    path = _ALGORITHM_DIR / "index.rst"
    if not path.exists() or path.read_text(encoding="utf-8") != index:
        path.write_text(index, encoding="utf-8")


def setup(app):
    app.connect("builder-inited", _generate_algorithm_pages)
