# Security Policy

## Supported versions

SCPP is a research library under active development. Security fixes are applied
to the latest released version on PyPI; there are no long-term support branches.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| Older releases | No — please upgrade |

## Reporting a vulnerability

Please **do not** open a public issue for a security problem.

Report it privately through GitHub's
[private vulnerability reporting](https://github.com/soft-clustering/soft-clustering/security/advisories/new)
form. If that is unavailable to you, email the maintainers at
kiyanrezaee17@gmail.com.

Please include:

- the affected version and how it was installed,
- a minimal reproducer,
- what an attacker gains, and under what assumptions.

You can expect an acknowledgement within 7 days and an assessment within 30
days. If the report is confirmed we will agree a disclosure timeline with you,
credit you in the advisory unless you prefer otherwise, and publish a fixed
release before the advisory becomes public.

## Scope

This is a numerical clustering library. It does not handle authentication,
process untrusted network input, or run as a privileged service, so the
realistic security surface is narrow. The following are in scope:

- code execution or file access triggered by loading a model, a dataset, or a
  benchmark result through the package's own APIs,
- a dependency we declare that ships a known vulnerability affecting our usage,
- unsafe deserialization anywhere in the package.

The following are **not** vulnerabilities in this project:

- resource exhaustion caused by passing a deliberately large `X`, a large
  `n_clusters`, or a high `max_iter` — these are caller-controlled,
- numerical instability, non-convergence, or poor clustering quality; please
  file those as ordinary bug reports,
- vulnerabilities in `scikit-learn`, `numpy`, `scipy`, `torch` or other
  dependencies that do not affect how this package uses them. Report those
  upstream.

## Datasets fetched at runtime

`soft_clustering.benchmarking.get_dataset` downloads OpenML datasets on first
use via `scikit-learn`'s `fetch_openml`. Those files come from a third party
and are cached by scikit-learn, not by this package. Treat them with the same
caution as any downloaded data; the bundled and synthetic dataset groups
require no network access at all.
