# Configuration file for the Sphinx documentation builder.

# -- Project information

project = "Soft Clustering"
copyright = "2025"
author = "University of Guilan"

release = "0.1"
version = "0.0.1"

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

# GitHub renders "#L123" line anchors client-side, so linkcheck cannot resolve
# them and reports every source-line citation in the algorithm pages as broken.
# Anchors are still verified for every other host.
linkcheck_anchors_ignore_for_url = [
    r"https://github\.com/.*",
]
# Publishers that serve 403 to automated clients. Their links are not broken —
# checking them only produces noise that hides the failures worth acting on.
# DOIs are included because they redirect to exactly these hosts.
linkcheck_ignore = [
    r"https://doi\.org/.*",
    r"https://dl\.acm\.org/.*",
    r"https://ieeexplore\.ieee\.org/.*",
    r"https://link\.springer\.com/.*",
    r"https://www\.sciencedirect\.com/.*",
    r"https://onlinelibrary\.wiley\.com/.*",
]
linkcheck_timeout = 15
linkcheck_retries = 2
# Unauthenticated GitHub requests are rate limited; waiting out a long backoff
# in CI is not worth it for a non-blocking job.
linkcheck_rate_limit_timeout = 30.0

# -- Options for HTML output

html_theme = "sphinx_rtd_theme"

# -- Options for EPUB output
epub_show_urls = "footnote"
