---
title: 'SCPP: soft clustering methods from several families behind one estimator protocol in Python'
tags:
  - Python
  - clustering
  - fuzzy clustering
  - possibilistic clustering
  - evidential clustering
  - overlapping community detection
  - topic models
authors:
  - name: Kiyan Rezaee
    corresponding: true
    affiliation: 1
  - name: Morteza Ziabakhsh
    affiliation: 1
  - name: Artin Bahrampour
    affiliation: 1
  - name: Seyed Mohammad Ghoreishi
    affiliation: 1
  - name: Asal Khajeh
    affiliation: 1
  - name: Ali Sajedifar
    affiliation: 1
  - name: Manny Chalak
    affiliation: 1
  - name: Ava Zerafatangiz
    affiliation: 1
  - name: Sadegh Eskandari
    affiliation: 1
affiliations:
  - name: Department of Computer Science, University of Guilan, Rasht, Iran
    index: 1
date: 8 October 2026
bibliography: paper.bib
---

# Summary

Clustering groups similar observations. *Soft* clustering methods do not
force each observation into exactly one group; they report a degree of
membership in every group instead. This is the natural description when a
document covers several topics, a person belongs to several social circles,
or a measurement lies between two regimes. Soft clustering has been developed
independently in several research communities, which produced fuzzy
[@ruspini1969new; @bezdek1981pattern], possibilistic
[@krishnapuram1993possibilistic], evidential [@masson2008ecm], probabilistic
[@dempster1977maximum], graph-based [@yang2013overlapping;
@psorakis2011overlapping; @airoldi2008mixed], topic-model
[@hofmann1999plsa; @blei2003latent] and consensus [@punera2008consensus]
formulations.

SCPP (Soft Clustering Python Package, distributed as `soft-clustering`)
implements 42 of these methods in one library. Every estimator is fitted on
the input its method requires: a feature matrix, a graph, a text corpus, an
image or a list of partitions to reconcile. After fitting, every estimator
exposes the same four attributes: a membership matrix `memberships_`, hard
labels `labels_`, prototypes `centers_` (when the method has them) and the
number of clusters produced, `n_clusters_`. Because the meaning of a
membership differs between families, each estimator also declares what its
output is (for example a fuzzy degree, a posterior probability, a typicality
or a pignistic probability) and whether its rows sum to one. SCPP also
records, for each estimator, how far its implementation has been verified.

# Statement of need

Researchers who apply soft clustering usually need to compare several
methods, often from different families, on their own data. Today this
requires installing one package per family and writing adapter code for
each: implementations differ in how the number of clusters is passed (the
implementations collected in SCPP originally used several different
parameter names for it), in whether the result is returned or stored, in
its orientation (samples-first or clusters-first) and in what it means. These differences
are easy to get wrong silently, for example by treating unnormalised
possibilistic typicalities as probabilities. SCPP removes this adapter layer
so that, within each input modality, exchanging one method for another is a
one-word change and evaluation code is written once.

SCPP is intended for researchers who use soft clustering as a tool (in data
analysis, software engineering, network science or text mining), for
methodologists who need baselines from several families, and for
contributors who want to add a method to a shared, tested code base.

# State of the field

Specialised packages offer more depth within one family than SCPP does.
evclust provides more evidential methods [@soubeiga2025evclustpaper];
fclust [@ferraro2015fclust] and FWSCA [@golzari2025feature] cover many
fuzzy c-means variants and validity indices; mclust [@scrucca2016mclust]
and Mixture-Models [@kasa2024mixture] cover mixture models; CDlib
[@rossetti2019cdlib] and Karate Club [@rozemberczki2020karateclub] provide
many overlapping community-detection methods; clue [@hornik2005clue]
implements consensus of soft partitions; and scikit-fuzzy
[@scikit_fuzzy_2024], fuzzy-c-means [@dias2019fuzzy], pyclustering
[@novikov2019pyclustering] and hdbscan [@mcinnes2017hdbscan] each provide
one or two soft methods. scikit-learn [@pedregosa2011scikit] offers
Gaussian mixtures and latent Dirichlet allocation under its estimator API
[@buitinck2013api], and ClustPy [@leiber2023benchmarking] offers deep and
subspace clustering under the same API.

The packages that provide a uniform interface also fix the input:
scikit-learn and ClustPy operate on feature matrices, CDlib and Karate Club
on graphs. Within each domain this is appropriate, but it means that a fuzzy
method for feature vectors, a graph affiliation model and a topic model
cannot be exchanged in one pipeline without an adapter. Among the packages
above we found none that exposes fuzzy, possibilistic, evidential,
probabilistic, graph, document and consensus methods through one object
model. SCPP addresses that gap; it does not aim to replace the specialised
packages, and its documentation points users to them.

# Software design

The central design decision is to standardise what an estimator produces,
not what it consumes. Forcing every method onto a feature matrix would hide
real differences (a topic model needs term counts, a consensus method needs
several partitions), whereas the fitted state can be described uniformly.

The protocol is implemented once, in `BaseSoftClusterer`. When an estimator
class is defined, `__init_subclass__` wraps its constructor and its own
`fit` or `fit_predict`. The constructor wrapper accepts `n_clusters` in
place of a method's historical parameter name and records the constructor
arguments, which makes `get_params`, `set_params` and scikit-learn's `clone`
work. The fit wrapper resets fitted state, calls the method's own code and
then locates and normalises its result: it transposes clusters-first
matrices, converts sparse output, encodes crisp consensus labels as one-hot
rows and derives `labels_`. An algorithm module therefore contains only the
mathematics of its method, and a new method needs no protocol code beyond
declaring where its result is stored. The cost is indirection: a
contributor must understand the wrapper when debugging.

Two further decisions follow from the differences between methods. First,
`predict(X)` on new data is available only for the 14 estimators whose
method defines an out-of-sample rule (mostly prototype-based ones); the
transductive estimators raise an error instead of silently returning the
training partition. Second, scikit-learn compatibility is claimed only where
it holds: 11 inductive feature-matrix estimators are tested with
scikit-learn's estimator checks, while estimators whose input is a graph,
a corpus or a list of partitions support parameter introspection and
cloning but not scikit-learn's `fit(X, y)` conventions.

A metadata registry is the single source of the algorithm table in the
README and documentation and of `list_algorithms()`. Each entry records the
method's family, input, output semantics, how the number of clusters is
obtained, its reference, and a verification status: *validated* (6
estimators agree with an independent implementation or reduce exactly to
one in a limit), *reference* (14 implement the published update equations,
checked against the publication), *adapted* (14 implement the method with
documented deviations) or *unverified* (8 follow the method as contributed;
only invariants and recovery of planted structure are tested). Stating
these levels makes the remaining verification work visible to users instead
of implying that every implementation is equally trustworthy.

Quality assurance is organised in layers. A conformance suite is
parameterised from the registry, so a new estimator is tested automatically
and none can be excluded: it checks the shape, sign and normalisation of
the output, consistency of labels with memberships, reproducibility under a
fixed `random_state`, isolation from NumPy's and PyTorch's global random
state, idempotent refitting and that inputs are not modified. Further
layers test recovery of planted structure for every input modality, edge
cases (duplicate points, singular covariances, extreme fuzzifiers,
non-finite input), agreement with scikit-fuzzy, scikit-learn and evclust,
invariance of the scale-free estimators to rescaling of the data, and each
method's own equations. The suite runs on Linux, macOS and Windows with Python 3.10 to 3.14 in
continuous integration, together with a job without the optional PyTorch
dependency. A benchmarking subpackage provides runtime, memory, scalability
and clustering-quality measurements with soft-clustering validity indices,
and scripts in the repository regenerate every committed benchmark table.

SCPP depends on NumPy [@harris2020array], SciPy [@virtanen2020scipy] and
scikit-learn; the four deep-learning estimators additionally require
PyTorch [@paszke2019pytorch] and are imported lazily, so the base
installation does not need it.

# Research impact statement

SCPP is used in research on decomposing monolithic software into
microservices. The Mo2oM approach [@ziabakhsh2025extracting]
formulates microservice extraction as overlapping (soft) clustering of a
program's classes, so that a class implementing a shared concern can belong
to several services. Its clustering stage has been integrated with SCPP so
that the clustering method is a parameter of the pipeline, and graph
estimators from SCPP (NOCD, BayesianNMF and BIGCLAM) have been exchanged in
it without changes to the downstream analysis. A preprint describes the
library and its validation in more detail [@rezaee2026scppPreprint]. SCPP
is distributed on PyPI under the MIT license, documented at
<https://soft-clustering.readthedocs.io>, and developed openly at
<https://github.com/soft-clustering/soft-clustering>, where contribution
guidelines, issue templates for bug reports, questions and algorithm
requests, and a guide to adding a method are provided.

[AUTHORS: before submission, replace or extend this paragraph with publicly
verifiable evidence of use, as JOSS requires: for example a public link to
the Mo2oM integration code, other publications or preprints that use SCPP,
or documented use by other groups. Do not submit with this note.]

# AI usage disclosure

[AUTHORS: complete and confirm.] Generative AI tools were used in the
development of this software and paper. Claude Code (Anthropic; model Claude
Opus 5.5) was used in October 2026 to audit the code and its numerical
behaviour, to make the stopping rules of several estimators independent of
the units of the data, to fix defects in KFCM, RPFKM, SFCMEP and WBSC, to
write the corresponding tests, and to draft this paper from the repository
and the authors' earlier manuscript. [AUTHORS: list any other tools, models and
versions used earlier in development, for example for code generation,
refactoring, tests, documentation or copy-editing, and where they were
used.] The human authors reviewed, edited and validated all AI-assisted
output, ran the test suite, and made the design decisions described above.

# Acknowledgements

[AUTHORS: acknowledge funding sources, or state that the work received no
specific funding, and acknowledge contributors who are not authors.]

# References
