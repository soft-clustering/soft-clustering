# EVCLUS (Evidential Clustering of Proximity Data)

## 🔍 Overview

EVCLUS assigns each object a **mass function** over a restricted frame of
focal sets — the `K` singletons plus the whole frame Ω, which carries the
ignorance — and fits those mass functions so that the *degree of conflict*
between two objects tracks their dissimilarity:

$$
J(M) \;=\; \frac{\sum_{i<j} (\kappa_{ij} - \delta_{ij})^2}
                 {\sum_{i<j} \delta_{ij}^2},
\qquad
\kappa_{ij} = \!\!\sum_{A \cap B = \emptyset}\!\! m_i(A)\, m_j(B).
$$

Unlike a probability vector, a mass function can say *"I do not know"*: an
object that sits between clusters puts mass on Ω rather than splitting it
evenly among the singletons. A uniform row `(1/3, 1/3, 1/3)` cannot be
distinguished from a genuinely ambiguous object; `m(Ω) = 0.9` can.

EVCLUS works directly on **proximities**, so it applies where no feature
vectors exist — only pairwise dissimilarities.

---

## ⚙️ Class Definition

```python
class EVCLUS(BaseSoftClusterer):
    def __init__(self, n_clusters: int = 3, max_iter: int = 200, n_init: int = 1,
                 tol: float = 1e-6, metric: str = "euclidean",
                 random_state: int | None = None):
        ...
```

---

## 📋 Parameters

| Parameter      | Type          | Default       | Description |
|----------------|---------------|---------------|-------------|
| `n_clusters`   | int           | 3             | Number of singleton focal sets |
| `max_iter`     | int           | 200           | L-BFGS-B iterations per restart |
| `n_init`       | int           | 1             | Random restarts; the lowest stress is kept |
| `tol`          | float         | 1e-6          | Optimiser tolerance |
| `metric`       | `"euclidean"` \| `"precomputed"` | `"euclidean"` | Whether `fit` receives features or an `(n, n)` dissimilarity matrix |
| `random_state` | int \| None   | None          | Seed for the restarts |

---

## 🚀 Usage

```python
import numpy as np
from soft_clustering import EVCLUS

rng = np.random.default_rng(0)
centres = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 5.0]])
X = np.vstack([rng.normal(c, 0.5, (40, 2)) for c in centres])

model = EVCLUS(n_clusters=3, n_init=3, random_state=0).fit(X)

masses = model.masses_          # (120, 4): three singletons + Omega
U = model.memberships_          # (120, 3) pignistic probabilities, rows sum to 1
ignorance = model.ignorance()   # (120,) mass on Omega
stress = model.stress_
```

On a dissimilarity matrix — the input EVCLUS was originally defined for:

```python
model = EVCLUS(n_clusters=3, metric="precomputed", random_state=0).fit(D)
```

---

### 📥 Input / 📤 Output

- **Input to `fit(X)`**: a feature matrix `(n_samples, n_features)`, or an
  `(n, n)` dissimilarity matrix when `metric="precomputed"`.
- **Fitted attributes**: `masses_` `(n, K+1)`, `memberships_` `(n, K)`,
  `labels_`, `stress_`, `n_clusters`.

---

## 🛠️ Methods

- `fit(X)` — minimise the stress and return `self`.
- `betp()` — pignistic probabilities, identical to `memberships_`.
- `ignorance()` — the mass each object puts on Ω.
- `predict()` / `predict_proba()` — the fitted partition. EVCLUS is
  transductive, so passing new data raises `NotImplementedError`.

---

## 📝 Implementation notes

**Closed-form conflict.** With singleton-plus-Ω focal sets the conflict has a
closed form, `κ_ij = s_i s_j − ⟨S_i, S_j⟩`, where `S` holds the singleton
masses and `s_i = 1 − m_i(Ω)`. Dissimilarities are rescaled to `[0, 1]`
because a conflict is a probability-like quantity.

**Optimisation.** The simplex constraint is handled by a softmax
reparameterisation rather than by projection, and the stress is minimised with
L-BFGS-B using the analytic gradient

$$
\frac{\partial J}{\partial S}
  = \frac{2}{C}\Big[ (G s)\mathbf{1}^\top - G S \Big],
\qquad G_{ij} = \kappa_{ij} - \delta_{ij}\;(i \neq j),
$$

which makes the fit deterministic given the initial parameters and avoids the
finite-difference cost of a gradient-free optimiser. The gradient is checked
against finite differences in `tests/test_evclus.py`.

**Degenerate input.** When every object is identical there is no structure to
fit, and the estimator returns total ignorance (`m(Ω) = 1` for every object),
whose pignistic transform is uniform.

---

### 📚 Reference

T. Denœux and M.-H. Masson. *EVCLUS: evidential clustering of proximity data.*
IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics),
34(1):95–109, 2004.
