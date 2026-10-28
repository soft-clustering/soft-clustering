API reference
=============

Estimators
----------

Every estimator is documented on its own :doc:`algorithm page
<algorithms/index>`. They all derive from :class:`BaseSoftClusterer`.

.. autoclass:: soft_clustering.BaseSoftClusterer
   :members: fit, fit_predict, predict, predict_proba, get_params, set_params

.. autofunction:: soft_clustering.list_algorithms

.. autoclass:: soft_clustering.AlgorithmInfo

.. autodata:: soft_clustering.DEEP_ESTIMATORS
   :annotation:

.. autoexception:: soft_clustering.NotFittedError

Validity indices
----------------

.. automodule:: soft_clustering.benchmarking.metrics
   :members:

Datasets
--------

.. automodule:: soft_clustering.benchmarking.datasets
   :members: get_dataset, available_datasets, available_groups, datasets_in_group, dataset_info, benchmark_suite
