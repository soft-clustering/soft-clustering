# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

# Configuration file for the Sphinx documentation builder.

# -- Project information

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_version

project = "Soft Clustering"
copyright = "2025-2026, the SCPP authors"
author = "Kiyan Rezaee and Morteza Ziabakhsh and Artin Bahrampour and Seyed Mohammad Ghoreishi and Asal Khaje and Ali Sajedifar and Manny Chalak and Ava Zerafatangiz and Sadegh Eskandari"

# Read from the installed distribution rather than restating it here, so the
# documentation cannot claim a version the package does not have.
try:
    release = _package_version("soft-clustering")
except PackageNotFoundError:  # building against a source tree, not an install
    release = "0.0.0+unknown"
version = ".".join(release.split(".")[:2])

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for the linkcheck builder

# Hosts whose failures say nothing about this project. Checking them produces
# noise that hides the failures actually worth acting on.
linkcheck_ignore = [
    # These pages are self-references to files in this repository, on the
    # branch the docs are built from. There are 26 of them, and GitHub rate
    # limits unauthenticated requests per source IP, so a CI runner on a shared
    # Actions address reliably collects 429s partway through the run — the
    # failures track how busy the runner's neighbours are, not whether the
    # links work. A rename that broke one of these would be visible in the
    # repository itself.
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
# Sphinx 8 default, set explicitly: a request that times out is reported with a
# "timeout" status rather than counted as a broken link. A slow host is not a
# rotted URL, and the distinction is the whole point of reading this report.
linkcheck_report_timeouts_as_broken = False

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
