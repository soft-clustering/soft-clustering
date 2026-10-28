SCPP: Soft Clustering Python Package
====================================

SCPP implements 42 soft clustering algorithms — fuzzy, possibilistic,
evidential, probabilistic, rough-set, kernel, graph, document, ensemble and
deep — behind one estimator protocol, together with a benchmarking suite for
comparing them on equal terms.

Hard clustering puts every sample in exactly one group. Soft clustering
assigns degrees of membership instead, which suits documents that cover
several topics, nodes in several communities, samples on a boundary and
outliers that belong nowhere. The methods that do this are spread across
decades of literature, each with its own interface and its own meaning of
"membership". SCPP gives them one interface, documents what each output
means, and states how far each implementation has been verified.

.. code-block:: bash

   pip install soft-clustering

.. code-block:: python

   import numpy as np
   from soft_clustering import FCM

   rng = np.random.default_rng(0)
   X = np.vstack([rng.normal([0, 0], 0.6, (100, 2)),
                  rng.normal([3, 3], 0.6, (100, 2))])

   model = FCM(n_clusters=2, random_state=0).fit(X)
   model.memberships_     # (200, 2) fuzzy memberships; rows sum to one
   model.labels_          # (200,)   arg-max hard assignment
   model.centers_         # (2, 2)   cluster prototypes

Start with the :doc:`user guide <overview>`, then browse the
:doc:`algorithms <algorithms/index>`.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   algorithms/index
   benchmarking
   api

Citing SCPP
-----------

Please cite the software (the repository's ``CITATION.cff``) and the
preprint `arXiv:2607.19620 <https://arxiv.org/abs/2607.19620>`_, as well as
the original publication of every algorithm you use; each algorithm page
gives it.

SCPP is distributed under the MIT license.
