Soft Clustering Python Package
=================================

Soft Clustering is a Python library for soft and fuzzy clustering algorithms. It provides a wide range of state-of-the-art methods for membership-based, possibilistic, evidential, graph, document, and deep clustering models.

This package is installable via `pip` and is designed to support research and applied workflows for clustering with uncertainty, overlapping membership, and soft assignments.

Official documentation is live and available at:

https://soft-clustering.readthedocs.io/en/latest/index.html


Features
--------

- Fuzzy clustering: `FCM`, `PCM`, `GK`, `SCM`, `SISC`, `KFCM`, `PFCM`, `CAFCM`, and related variants
- Probabilistic and mixture models: `GMM`, `LDA`, `PLSI`, `MBMM`, `MMSB`, `BGMM`
- Rough and possibilistic clustering: `RoughKMeans`, `WBSC`, `NOCD`, `DMon`, `RDFKC`
- Nonnegative matrix factorization and graph clustering: `BayesianNMF`, `BIGCLAM`, `CDCgS`, `SoftDBSCANGM`
- Document and graph clustering methods: `KMART`, `SCSPA` and additional soft clustering algorithms


Installation
------------

Install from PyPI:

.. code-block:: console

   pip install soft-clustering


Quick Start
-----------

Basic use with the package API:

.. code-block:: python

   import numpy as np
   from soft_clustering import FCM

   X = np.array([[1.0, 2.0], [1.1, 2.1], [8.0, 8.1], [8.2, 7.9]])
   model = FCM(m=2.0, max_iter=150, tol=1e-5)
   memberships = model.fit_predict(X, K=2)
   centers = model.centers_

   print("Soft membership shape:", memberships.shape)
   print("Cluster centers:", centers)


Documentation
-------------

Full API reference, examples, and algorithm descriptions are available in the official documentation:

https://soft-clustering.readthedocs.io/en/latest/index.html


Research & Citation
-------------------

This package is part of an academic research project submitted to the JMLR MLOSS journal. A publication is forthcoming.

- Paper link: *TBD*
- Citation: *TBD*


Contributing
------------

Contributions, bug reports, and feature requests are welcome. Please open issues or pull requests on the repository.


License
-------

Distributed under the terms of the MIT license.
