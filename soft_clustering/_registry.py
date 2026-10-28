# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Metadata for every estimator: the single source of the algorithm tables.

The README table, the documentation pages and :func:`list_algorithms` are all
generated from :data:`ALGORITHMS`, and ``tests/test_registry.py`` checks that
it stays in step with the package's exports and with each class's declared
behaviour, so the published description of an estimator cannot drift from the
code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: What ``memberships_`` means for each output code.
OUTPUT_KINDS: dict[str, str] = {
    "F": "fuzzy membership (rows sum to one)",
    "T": "possibilistic typicality (rows need not sum to one)",
    "P": "posterior probability (rows sum to one)",
    "E": "pignistic probability of a credal partition (rows sum to one)",
    "R": "rough-set membership read off lower/upper approximations",
    "A": "non-negative affiliation strength (unnormalised)",
    "M": "mixed-membership or topic proportion (rows sum to one)",
    "S": "soft assignment from a relaxation (rows sum to one)",
    "D": "binary degree of match; a sample may match several clusters or none",
    "H": "crisp consensus partition, one-hot",
}

#: How the number of clusters is obtained.
CLUSTER_COUNT: dict[str, str] = {
    "given": "given by the user",
    "discovered": "discovered by the method",
    "selected": "selected by an internal validity-index sweep",
}

#: Verification levels, from strongest to weakest.
STATUS: dict[str, str] = {
    "validated": (
        "agrees with an independent implementation, or reduces exactly to one "
        "in a limit, in tests/test_external_agreement.py"
    ),
    "reference": (
        "implements the update equations of the cited publication, checked "
        "against it in the 0.2.0 audit; invariants and planted-structure "
        "recovery are tested, but no independent implementation was compared"
    ),
    "adapted": (
        "implements the cited method with documented deviations; see the "
        "Notes or 'Deviation' paragraphs of the estimator's docstring"
    ),
    "unverified": (
        "follows the cited method as implemented by contributors; the "
        "equations have not been checked against the publication. Recovery of "
        "planted structure is tested, nothing more"
    ),
}


@dataclass(frozen=True)
class AlgorithmInfo:
    """Description of one SCPP estimator."""

    name: str
    method: str
    family: str
    input: str
    output: str
    n_clusters: str
    status: str
    reference: str
    link: str = ""
    note: str = ""

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


_A = AlgorithmInfo

ALGORITHMS: dict[str, AlgorithmInfo] = {
    a.name: a
    for a in [
        # ---------------------------------------------------------- fuzzy
        _A(
            "FCM",
            "Fuzzy c-means",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "validated",
            "Bezdek (1981)",
            "https://doi.org/10.1007/978-1-4757-0450-1",
            "agrees with scikit-fuzzy",
        ),
        _A(
            "GK",
            "Gustafson-Kessel fuzzy clustering",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "validated",
            "Gustafson and Kessel (1979)",
            "https://doi.org/10.1109/CDC.1978.268028",
            "reduces to FCM on isotropic clusters",
        ),
        _A(
            "GathGeva",
            "Gath-Geva fuzzy maximum-likelihood clustering",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "reference",
            "Gath and Geva (1989)",
            "https://doi.org/10.1016/0167-8655(89)90040-8",
        ),
        _A(
            "ENTROPYFCM",
            "Entropy-regularised fuzzy c-means",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "reference",
            "Li and Mukaidono (1995)",
            "",
            "the paper cites Gupta et al. (2021), which is not implemented",
        ),
        _A(
            "CAFCM",
            "Collaborative annealing fuzzy c-means",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "adapted",
            "Li and Wang (2024)",
            "https://doi.org/10.1109/TFUZZ.2023.3319663",
            "meta-heuristic re-initialisation rule is SCPP's own",
        ),
        _A(
            "CAFHFCM",
            "Centroid auto-fused hierarchical fuzzy c-means",
            "fuzzy",
            "feature matrix",
            "F",
            "discovered",
            "adapted",
            "Lin and Chen (2021)",
            "https://doi.org/10.1109/TFUZZ.2020.2991306",
            "MM instead of ADMM; warm-started path on a fixed alpha grid",
        ),
        _A(
            "AFCM",
            "Adaptive fuzzy c-means with graph embedding",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "reference",
            "Chen et al. (2024)",
            "https://arxiv.org/abs/2405.13427",
            "O(n^3) per iteration",
        ),
        _A(
            "AFCMSimple",
            "Adaptive fuzzy c-means without graph embedding",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "reference",
            "Chen et al. (2024)",
            "https://arxiv.org/abs/2405.13427",
        ),
        _A(
            "RPFKM",
            "Robust projected fuzzy k-means",
            "fuzzy",
            "feature matrix",
            "F",
            "given",
            "unverified",
            "Zhao et al. (2022)",
            "https://doi.org/10.1016/j.neucom.2022.03.043",
        ),
        _A(
            "FCC",
            "Fuzzy color clustering",
            "fuzzy",
            "feature matrix (colors)",
            "F",
            "given",
            "adapted",
            "Kim and Lee (2024)",
            "https://arxiv.org/abs/2407.06782",
            "k-means++ instead of reference-color initialisation",
        ),
        _A(
            "SFCMEP",
            "Semi-supervised fuzzy clustering with a prior membership matrix",
            "fuzzy, semi-supervised",
            "feature matrix + partial labels",
            "F",
            "given",
            "unverified",
            "Xu et al. (2024)",
            "https://doi.org/10.1016/j.eswa.2023.121812",
        ),
        _A(
            "FeMIFuzzy",
            "Federated fuzzy clustering with multiple imputation",
            "fuzzy, federated",
            "per-client feature matrices",
            "F",
            "selected",
            "adapted",
            "Ngo et al. (2024)",
            "https://doi.org/10.1109/JIOT.2023.3343719",
        ),
        _A(
            "AFCMAdaptive",
            "Adaptive fuzzy c-means for images with a bias field",
            "fuzzy, image",
            "grayscale image",
            "F",
            "given",
            "reference",
            "Pham and Prince (1999)",
            "https://doi.org/10.1016/S0167-8655(98)00121-4",
        ),
        _A(
            "RDFKC",
            "Robust deep fuzzy k-means",
            "fuzzy, deep",
            "feature matrix or image tensor",
            "F",
            "given",
            "unverified",
            "Wu et al. (2024)",
            "https://doi.org/10.1016/j.patcog.2024.110504",
        ),
        # --------------------------------------------------------- kernel
        _A(
            "KFCM",
            "Kernelised fuzzy c-means",
            "kernel",
            "feature matrix",
            "F",
            "given",
            "reference",
            "Zhang and Chen (2004)",
            "https://doi.org/10.1016/j.artmed.2004.01.012",
        ),
        _A(
            "SKFCM",
            "Spatially constrained kernelised fuzzy c-means",
            "kernel, image",
            "pixel features + image shape",
            "F",
            "given",
            "reference",
            "Zhang and Chen (2004)",
            "https://doi.org/10.1016/j.artmed.2004.01.012",
        ),
        _A(
            "KFCCL",
            "Kernel-based fuzzy competitive learning",
            "kernel",
            "feature matrix",
            "F",
            "given",
            "unverified",
            "Mizutani and Miyamoto (2005)",
            "https://doi.org/10.1109/FUZZY.2005.1452468",
            "O(n^2) Gram matrix up to 5000 samples; O(Kn) memory above",
        ),
        _A(
            "SoftKSC",
            "Soft kernel spectral clustering",
            "kernel, spectral",
            "feature matrix",
            "S",
            "given",
            "reference",
            "Langone et al. (2013)",
            "https://doi.org/10.1109/IJCNN.2013.6706850",
            "O(n^3) training",
        ),
        # -------------------------------------------- possibilistic, evidential
        _A(
            "PCM",
            "Possibilistic c-means",
            "possibilistic",
            "feature matrix",
            "T",
            "given",
            "reference",
            "Krishnapuram and Keller (1993)",
            "https://doi.org/10.1109/91.227387",
        ),
        _A(
            "PFCM",
            "Possibilistic fuzzy c-means",
            "possibilistic",
            "feature matrix",
            "F",
            "given",
            "validated",
            "Pal et al. (2005)",
            "https://doi.org/10.1109/TFUZZ.2004.840099",
            "reduces to FCM at zero typicality weight",
        ),
        _A(
            "ECM",
            "Evidential c-means",
            "evidential",
            "feature matrix",
            "E",
            "given",
            "validated",
            "Masson and Denoeux (2008)",
            "https://doi.org/10.1016/j.patcog.2007.08.014",
            "agrees with evclust (the authors' package)",
        ),
        _A(
            "EVCLUS",
            "Evidential clustering of proximity data",
            "evidential",
            "feature matrix or dissimilarities",
            "E",
            "given",
            "adapted",
            "Denoeux and Masson (2004)",
            "https://doi.org/10.1109/TSMCB.2002.806496",
            "focal sets restricted to singletons and the whole frame; " "O(n^2) memory",
        ),
        # ------------------------------------------------ probabilistic, rough
        _A(
            "GMM",
            "Gaussian mixture model (EM)",
            "probabilistic",
            "feature matrix",
            "P",
            "given",
            "validated",
            "Dempster et al. (1977)",
            "https://doi.org/10.1111/j.2517-6161.1977.tb01600.x",
            "agrees with scikit-learn",
        ),
        _A(
            "MBMM",
            "Mixture of Beta distributions",
            "probabilistic",
            "feature matrix in (0, 1)",
            "P",
            "given",
            "adapted",
            "Hsu and Chen (2024)",
            "https://doi.org/10.1007/978-981-97-2242-6_19",
            "independent Beta marginals; moment-matching M-step",
        ),
        _A(
            "BGMM",
            "Beta-Gaussian mixture over two views",
            "probabilistic, multi-view",
            "a Gaussian and a Beta view",
            "P",
            "given",
            "adapted",
            "Dai et al. (2008)",
            "",
            "moment-matching M-step for the Beta view",
        ),
        _A(
            "SoftDBSCANGM",
            "Soft DBSCAN with Gaussian-mixture refinement",
            "density",
            "feature matrix",
            "F",
            "discovered",
            "unverified",
            "Smiti and Elouedi (2016)",
            "https://doi.org/10.1109/IS.2016.7737459",
            "every DBSCAN noise point becomes its own component",
        ),
        _A(
            "SCM",
            "Subtractive clustering",
            "density",
            "feature matrix",
            "F",
            "discovered",
            "reference",
            "Chiu (1994)",
            "https://doi.org/10.3233/IFS-1994-2306",
        ),
        _A(
            "RoughKMeans",
            "Rough k-means",
            "rough set",
            "feature matrix",
            "R",
            "given",
            "reference",
            "Lingras and West (2004)",
            "https://doi.org/10.1023/B:JIIS.0000029668.88665.1a",
            "relative boundary threshold of Peters (2006)",
        ),
        # ----------------------------------------------------------- graphs
        _A(
            "BIGCLAM",
            "Cluster affiliation model for big networks",
            "graph",
            "adjacency matrix",
            "A",
            "given",
            "adapted",
            "Yang and Leskovec (2013)",
            "https://doi.org/10.1145/2433396.2433471",
            "random initialisation",
        ),
        _A(
            "BayesianNMF",
            "Bayesian NMF for overlapping communities",
            "graph",
            "adjacency matrix",
            "M",
            "given",
            "reference",
            "Psorakis et al. (2011)",
            "https://doi.org/10.1103/PhysRevE.83.066114",
            "n_clusters is an upper bound; the attribute n_effective reports the pruned count",
        ),
        _A(
            "MMSB",
            "Mixed-membership stochastic blockmodel",
            "graph",
            "adjacency matrix",
            "M",
            "given",
            "reference",
            "Airoldi et al. (2008)",
            "https://jmlr.org/papers/v9/airoldi08a.html",
            "O(n^2 K) memory",
        ),
        _A(
            "NOCD",
            "Neural overlapping community detection",
            "graph, deep",
            "adjacency matrix (+ node features)",
            "A",
            "given",
            "adapted",
            "Shchur and Gunnemann (2019)",
            "https://arxiv.org/abs/1909.12201",
        ),
        _A(
            "DMoN",
            "Deep modularity networks",
            "graph, deep",
            "node features + graph",
            "S",
            "given",
            "adapted",
            "Tsitsulin et al. (2023)",
            "https://jmlr.org/papers/v24/20-998.html",
            "two-layer GCN instead of the reference architecture",
        ),
        _A(
            "CDCGS",
            "Community detection via Gumbel-softmax",
            "graph, deep",
            "adjacency matrix",
            "S",
            "given",
            "adapted",
            "Acharya and Zhang (2020)",
            "https://doi.org/10.1007/s42979-020-00264-2",
            "graph preprocessing stage not reproduced",
        ),
        # -------------------------------------------------------- documents
        _A(
            "LDA",
            "Latent Dirichlet allocation",
            "topic model",
            "documents or term counts",
            "M",
            "given",
            "validated",
            "Blei et al. (2003)",
            "https://jmlr.org/papers/v3/blei03a.html",
            "agrees with scikit-learn",
        ),
        _A(
            "PLSI",
            "Probabilistic latent semantic indexing",
            "topic model",
            "documents or term counts",
            "M",
            "given",
            "reference",
            "Hofmann (1999)",
            "",
        ),
        _A(
            "SISC",
            "Similarity-based soft clustering of documents",
            "document",
            "documents",
            "D",
            "discovered",
            "unverified",
            "Lin and Kondadadi (2001a)",
            "https://doi.org/10.1109/DASFAA.2001.916362",
        ),
        _A(
            "WBSC",
            "Word-based soft clustering of documents",
            "document",
            "documents",
            "D",
            "discovered",
            "unverified",
            "Lin and Kondadadi (2001b)",
            "",
        ),
        _A(
            "KMART",
            "Modified fuzzy ART for documents",
            "document",
            "documents",
            "D",
            "discovered",
            "unverified",
            "Kondadadi and Kozma (2002)",
            "https://doi.org/10.1109/IJCNN.2002.1007544",
        ),
        # --------------------------------------------------------- ensembles
        _A(
            "SCSPA",
            "Soft CSPA consensus",
            "ensemble",
            "list of membership matrices",
            "H",
            "given",
            "adapted",
            "Punera and Ghosh (2008)",
            "https://doi.org/10.1080/08839510802170546",
            "k-means instead of METIS",
        ),
        _A(
            "SHBGF",
            "Soft HBGF consensus",
            "ensemble",
            "list of membership matrices",
            "H",
            "given",
            "adapted",
            "Punera and Ghosh (2008)",
            "https://doi.org/10.1080/08839510802170546",
            "k-means instead of METIS",
        ),
        _A(
            "SMCLA",
            "Soft MCLA consensus",
            "ensemble",
            "list of membership matrices",
            "F",
            "given",
            "adapted",
            "Punera and Ghosh (2008)",
            "https://doi.org/10.1080/08839510802170546",
            "k-means instead of METIS",
        ),
    ]
}


def list_algorithms(
    family: str | None = None, status: str | None = None
) -> list[AlgorithmInfo]:
    """Describe the estimators SCPP provides.

    Parameters
    ----------
    family : str, optional
        Keep only estimators whose family contains this word (for example
        ``"graph"``, ``"fuzzy"`` or ``"deep"``).
    status : {"validated", "reference", "adapted", "unverified"}, optional
        Keep only estimators with this verification status.

    Returns
    -------
    list of AlgorithmInfo
        One record per estimator, ordered by name. See :data:`OUTPUT_KINDS`,
        :data:`CLUSTER_COUNT` and :data:`STATUS` for the meaning of the coded
        fields.

    Examples
    --------
    >>> from soft_clustering import list_algorithms
    >>> [a.name for a in list_algorithms(family="evidential")]
    ['ECM', 'EVCLUS']
    """
    if status is not None and status not in STATUS:
        raise ValueError(f"status must be one of {sorted(STATUS)}; got {status!r}.")
    items = sorted(ALGORITHMS.values(), key=lambda a: a.name.lower())
    if family is not None:
        items = [a for a in items if family.lower() in a.family.lower()]
    if status is not None:
        items = [a for a in items if a.status == status]
    return items
