Welcome to soft-clustering documentation!
===============================================


.. image:: https://raw.githubusercontent.com/soft-clustering/soft-clustering/refs/heads/main/SCPP_Poster.png


Soft Clustering is a comprehensive Python library for soft and fuzzy clustering algorithms. It provides a wide range of state-of-the-art methods for membership-based, possibilistic, evidential, graph, document, and deep clustering models.

This package is designed to support research and applied workflows involving clustering under uncertainty, overlapping memberships, and soft assignments.


Features
--------

- **Fuzzy clustering**: `FCM`, `PCM`, `GK`, `SCM`, `SISC`, `KFCM`, `PFCM`, `CAFCM`, and related variants
- **Probabilistic and mixture models**: `GMM`, `LDA`, `PLSI`, `MBMM`, `MMSB`, `BGMM`
- **Rough and possibilistic clustering**: `RoughKMeans`, `WBSC`, `NOCD`, `DMon`, `RDFKC`
- **Nonnegative matrix factorization and graph clustering**: `BayesianNMF`, `BIGCLAM`, `CDCgS`, `SoftDBSCANGM`
- **Document and graph clustering methods**: `KMART`, `SCSPA` and additional soft clustering algorithms


Installation
------------

The complete package, including deep learning-based algorithms, can be installed from PyPI using:

.. code-block:: bash

   pip install soft-clustering[deep]

For users who only need the core algorithms (without deep learning dependencies), install the base version:

.. code-block:: bash

   pip install soft-clustering


Quick Start
-----------

Basic usage with the package API:

.. code-block:: python

   import numpy as np
   from soft_clustering import FCM

   X = np.array([[1.0, 2.0], [1.1, 2.1], [8.0, 8.1], [8.2, 7.9]])
   model = FCM(m=2.0, max_iter=150, tol=1e-5)
   memberships = model.fit_predict(X, K=2)
   centers = model.centers_

   print("Soft membership shape:", memberships.shape)
   print("Cluster centers:", centers)


Testing
-------

The project includes a comprehensive test suite in the `tests/`_ directory, covering all implemented algorithms.

To run the tests:

.. code-block:: bash

   pip install -e ".[deep,dev]"   # Install in development mode
   pytest

See `tests/HOW_TO_RUN.txt`_ for more details.


.. _`tests/`: https://github.com/soft-clustering/soft-clustering/tree/main/tests
.. _`tests/HOW_TO_RUN.txt`: https://github.com/soft-clustering/soft-clustering/blob/main/tests/HOW_TO_RUN.txt


Research & Citation
-------------------

This package is part of an academic research project submitted to the JMLR MLOSS journal. A publication is forthcoming.

- Paper link: https://arxiv.org/abs/2607.19620
- Citation:

.. code-block:: plaintext

   @misc{rezaee2026scppunifiedpythonlibrary,
      title={SCPP: A Unified Python Library for Soft Clustering}, 
      author={Kiyan Rezaee and Morteza Ziabakhsh and Artin Bahrampour and Seyed Mohammad Ghoreishi and Asal Khaje and Ali Sajedifar and Manny Chalak and Ava Zerafatangiz and Sadegh Eskandari},
      year={2026},
      eprint={2607.19620},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.19620}, 
   }


Contributing
------------

Contributions, bug reports, and feature requests are welcome. Please open issues or pull requests on the repository.

Please see our:

- `Contributing Guidelines`_
- `Code of Conduct`_

.. _`Contributing Guidelines`: https://github.com/soft-clustering/soft-clustering/blob/main/CONTRIBUTING.md
.. _`Code of Conduct`: https://github.com/soft-clustering/soft-clustering/blob/main/CODE_OF_CONDUCT.md


License
-------

Distributed under the terms of the MIT license.


.. toctree::
   :maxdepth: 2

   nocd
   pcm
   kfccl
   rpfkm
   bgmm
   scm
   wbsc
   afcmSimple
   cafhfcm
   rdfkc
   softksc
   EntropyFCM
   gmm
   mmsb
   bnmf
   shbgf
   rough_k_means
   femifuzzy
   softdbscangm
   kmart
   fcc
   scspa
   ecm
   dmon
   sisc
   cdcgs
   bigclam
   cafcm
   fcm
   pfcm
   gk
   kfcm
   afcmadaptive
   afcm
   skfcm
   plsi
   smcla
   lda
   mbmm
