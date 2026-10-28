# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Planted-structure recovery for every estimator.

The conformance suite (``test_protocol.py``) checks that an estimator's
output has the right *form*. It cannot tell whether the method works: before
this module existed, five estimators passed every conformance check while
being broken --- BGMM returned exactly 1/K for every sample, AFCMAdaptive
segmented a two-level image with an adjusted Rand index of 0, RDFKC published
a one-hot of its labels as its "soft" output, ENTROPYFCM ignored its entropy
weight, and SoftKSC implemented a different method.

Here every estimator is fitted on an easy instance of its input modality with
known ground truth, and its hard partition must agree with the truth. The
instances are deliberately easy, so a failure means the method or its
implementation is broken, not that the data is hard. Thresholds were set from
measured scores, recorded next to each case, with a margin for platform
differences.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

import soft_clustering as sc

HAVE_DEEP = importlib.util.find_spec("torch") is not None


def blobs(k=3, n_per=40, d=2, spread=0.5, sep=5.0, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(0, sep, (k, d))
    # Push centres apart so the instance is unambiguous.
    for _ in range(50):
        diff = centres[:, None, :] - centres[None, :, :]
        dist = np.linalg.norm(diff, axis=2) + np.eye(k) * 1e9
        if dist.min() > sep:
            break
        centres += 0.5 * (diff / dist[:, :, None] ** 2).sum(axis=1) * sep
    X = np.vstack([rng.normal(c, spread, (n_per, d)) for c in centres])
    return X, np.repeat(np.arange(k), n_per)


def planted_graph(n=90, k=3, p_in=0.5, p_out=0.03, seed=1):
    rng = np.random.default_rng(seed)
    z = np.repeat(np.arange(k), n // k)
    P = np.full((k, k), p_out)
    np.fill_diagonal(P, p_in)
    A = (rng.random((n, n)) < P[z][:, z]).astype(float)
    A = np.triu(A, 1)
    return A + A.T, z


def corpus(per=10):
    topics = [
        "fuzzy clustering membership degree centroid partition prototype",
        "graph network community detection nodes edges modularity",
        "topic document word text corpus language vocabulary",
    ]
    rng = np.random.default_rng(3)
    docs, truth = [], []
    for t, words in enumerate(topics):
        vocab = words.split()
        for _ in range(per):
            docs.append(" ".join(rng.choice(vocab, size=6, replace=True)))
            truth.append(t)
    return docs, np.array(truth)


def score(model, truth):
    return adjusted_rand_score(truth, model.labels_)


# Each case: estimator name -> (fit callable returning (model, truth), threshold)
# The measured score on the reference platform is given in the comment.


def _feature(name, threshold, data=None, **kwargs):
    def run():
        X, y = data() if data else blobs()
        kwargs.setdefault("random_state", 0)
        return getattr(sc, name)(n_clusters=3, **kwargs).fit(X), y

    return run, threshold


def _mbmm():
    rng = np.random.default_rng(0)
    parts = [rng.beta(a, b, (60, 3)) for a, b in ((2, 8), (8, 2), (20, 20))]
    X = np.clip(np.vstack(parts), 1e-3, 1 - 1e-3)
    return sc.MBMM(n_clusters=3, random_state=0).fit(X), np.repeat([0, 1, 2], 60)


def _sfcmep():
    X, y = blobs()
    labels = np.full(y.size, -1)
    labels[[0, 1, 40, 41, 80, 81]] = y[[0, 1, 40, 41, 80, 81]]
    return sc.SFCMEP(n_clusters=3, random_state=0).fit(X, labels), y


def _bgmm():
    rng = np.random.default_rng(0)
    g = np.concatenate([rng.normal(-2, 0.5, 80), rng.normal(2, 0.5, 80)])
    b = np.clip(
        np.concatenate([rng.beta(2, 6, 80), rng.beta(6, 2, 80)]), 1e-3, 1 - 1e-3
    )
    return sc.BGMM(n_clusters=2, random_state=0).fit(g, b), np.repeat([0, 1], 80)


def _image(name, **kwargs):
    def run():
        rng = np.random.default_rng(0)
        truth = np.zeros((24, 24), dtype=int)
        truth[:, 8:16] = 1
        truth[:, 16:] = 2
        yy, xx = np.mgrid[0:24, 0:24] / 24.0
        gain = 1.0 + 0.3 * (xx - 0.5)
        image = np.array([0.2, 0.5, 0.8])[truth] * gain + rng.normal(0, 0.02, (24, 24))
        if name == "AFCMAdaptive":
            model = sc.AFCMAdaptive(n_clusters=3, **kwargs).fit(image)
        else:
            model = sc.SKFCM(n_clusters=3, random_state=0, **kwargs).fit(
                image.reshape(-1, 1), (24, 24)
            )
        return model, truth.ravel()

    return run


def _femifuzzy():
    rng = np.random.default_rng(11)
    names = ["a", "b", "c"]
    first = np.vstack([rng.normal(-3, 0.4, (15, 3)), rng.normal(3, 0.4, (15, 3))])
    second = np.vstack([rng.normal(3, 0.4, (12, 3)), rng.normal(-3, 0.4, (12, 3))])
    first[0, 1] = np.nan  # missing values are part of the method's remit
    truth = np.concatenate([np.repeat([0, 1], 15), np.repeat([1, 0], 12)])
    return sc.FeMIFuzzy(random_state=0).fit([first, second], [names, names]), truth


def _graph(name, **kwargs):
    def run():
        A, z = planted_graph()
        return getattr(sc, name)(n_clusters=3, random_state=0, **kwargs).fit(A), z

    return run


def _dmon():
    A, z = planted_graph()
    model = sc.DMoN(
        in_channels=A.shape[0], hidden_channels=16, n_clusters=3, random_state=0
    )
    return model.fit(np.eye(A.shape[0]), adj=A), z


def _nocd():
    import scipy.sparse as sp

    A, z = planted_graph()
    model = sc.NOCD(n_clusters=3, random_state=0, hidden_sizes=[32], batch_size=2000)
    model.fit(sp.csr_matrix(A))
    return model, z


def _rdfkc():
    import torch.nn as nn

    X, y = blobs(d=8)
    enc = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 3))
    dec = nn.Sequential(nn.Linear(3, 16), nn.ReLU(), nn.Linear(16, 8))
    model = sc.RDFKC(
        n_clusters=3, encoder=enc, decoder=dec, random_state=0, max_iter=40, lr=1e-3
    )
    return model.fit(X.astype("float32")), y


def _documents(name, **kwargs):
    def run():
        docs, truth = corpus()
        return getattr(sc, name)(**kwargs).fit(docs), truth

    return run


def _ensemble(name):
    def run():
        rng = np.random.default_rng(0)
        truth = np.repeat([0, 1, 2], 25)
        ensemble = []
        for _ in range(4):
            perm = rng.permutation(3)
            U = rng.random((75, 3)) * 0.3
            U[np.arange(75), perm[truth]] += 1.0
            ensemble.append(U / U.sum(axis=1, keepdims=True))
        return getattr(sc, name)(n_clusters=3, random_state=0).fit(ensemble), truth

    return run


CASES = {
    # --- feature matrix ------------------------------------------------------
    "FCM": _feature("FCM", 0.95),
    "GK": _feature("GK", 0.95),
    "GMM": _feature("GMM", 0.95),
    "GathGeva": _feature("GathGeva", 0.95),
    "PCM": _feature("PCM", 0.95),
    "PFCM": _feature("PFCM", 0.95),
    "ECM": _feature("ECM", 0.95),
    "EVCLUS": _feature("EVCLUS", 0.95),
    "ENTROPYFCM": _feature("ENTROPYFCM", 0.95),
    "KFCM": _feature("KFCM", 0.95, sigma=4.0),
    "KFCCL": _feature("KFCCL", 0.95, gamma=0.05, lambda_=3.0),
    "CAFCM": _feature("CAFCM", 0.95),
    "AFCM": _feature("AFCM", 0.95),
    "AFCMSimple": _feature("AFCMSimple", 0.95),
    "RPFKM": _feature("RPFKM", 0.9, d=2),
    "FCC": _feature("FCC", 0.95, jnd=2.0),  # jnd ~ 4 sd of a cluster
    "RoughKMeans": _feature("RoughKMeans", 0.95),
    "SoftKSC": (
        lambda: (sc.SoftKSC(n_clusters=3, gamma=0.2).fit(blobs()[0]), blobs()[1]),
        0.95,
    ),
    # The default alpha="auto" also has to discover K = 3.
    "CAFHFCM": (lambda: (sc.CAFHFCM(random_state=0).fit(blobs()[0]), blobs()[1]), 0.95),
    "SCM": (lambda: (sc.SCM().fit(blobs()[0]), blobs()[1]), 0.95),
    "SoftDBSCANGM": (
        lambda: (sc.SoftDBSCANGM(eps=1.0).fit(blobs()[0]), blobs()[1]),
        0.95,
    ),
    "MBMM": (_mbmm, 0.9),
    # --- feature matrix with additional structure ----------------------------
    "SFCMEP": (_sfcmep, 0.95),
    "BGMM": (_bgmm, 0.95),
    "FeMIFuzzy": (_femifuzzy, 0.95),
    "AFCMAdaptive": (_image("AFCMAdaptive"), 0.95),
    "SKFCM": (_image("SKFCM", gamma=20.0), 0.95),
    # --- graphs ----------------------------------------------------------------
    "BIGCLAM": (_graph("BIGCLAM"), 0.7),
    "BayesianNMF": (_graph("BayesianNMF"), 0.95),
    "MMSB": (_graph("MMSB"), 0.95),
    "CDCGS": (_graph("CDCGS"), 0.95),
    "DMoN": (_dmon, 0.95),
    "NOCD": (_nocd, 0.9),
    # --- documents -----------------------------------------------------------
    "LDA": (_documents("LDA", n_clusters=3, random_state=0), 0.85),  # 0.898
    "PLSI": (_documents("PLSI", n_clusters=3, random_state=0), 0.9),
    # Methods that discover the number of clusters may split a topic, which
    # ARI penalises; they are scored by purity in test_discovered_clusters_are_pure.
    "WBSC": (_documents("WBSC", min_doc_freq=2, max_doc_freq_ratio=0.4), 0.95),
    # --- ensembles -----------------------------------------------------------
    "SCSPA": (_ensemble("SCSPA"), 0.95),
    "SHBGF": (_ensemble("SHBGF"), 0.95),
    "SMCLA": (_ensemble("SMCLA"), 0.95),
    # --- deep ------------------------------------------------------------------
    "RDFKC": (_rdfkc, 0.9),
}


#: Discovered-K document methods, scored by homogeneity (every discovered
#: cluster holds a single topic) and by the fraction of documents assigned.
PURITY_CASES = {
    "KMART": ({"vigilance_param": 0.3}, 1.0),  # 12 pure clusters, all assigned
    "SISC": ({"n_clusters": 3, "random_state": 0}, 0.5),  # lambda leaves ~half out
}


def test_every_estimator_has_a_recovery_case():
    names = set(sc._ESTIMATORS)
    covered = set(CASES) | set(PURITY_CASES)
    assert names == covered, sorted(names ^ covered)


@pytest.mark.parametrize("name", sorted(PURITY_CASES))
def test_discovered_clusters_are_pure(name):
    from sklearn.metrics import homogeneity_score

    kwargs, min_assigned = PURITY_CASES[name]
    docs, truth = corpus()
    model = getattr(sc, name)(**kwargs).fit(docs)
    assigned = model.labels_ >= 0
    assert assigned.mean() >= min_assigned
    assert homogeneity_score(truth[assigned], model.labels_[assigned]) > 0.95


@pytest.mark.parametrize("name", sorted(CASES))
def test_recovers_planted_structure(name):
    if name in sc.DEEP_ESTIMATORS and not HAVE_DEEP:
        pytest.skip("needs the optional `deep` extra")
    run, threshold = CASES[name]
    model, truth = run()
    ari = score(model, truth)
    assert ari >= threshold, f"{name}: adjusted Rand index {ari:.3f} < {threshold}"
