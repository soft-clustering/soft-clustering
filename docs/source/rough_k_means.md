# Rough K-Means (RoughKMeans)

> **A clustering method where points can belong to multiple clusters, and centers are updated using both definite and possible members.**

---

## Overview

`RoughKMeans` is a clustering algorithm based on rough-set theory. Instead of assigning each sample to exactly one cluster, it defines:

- **Lower approximation (L)**: samples that definitely belong to a cluster.
- **Upper approximation (U)**: samples that possibly belong to a cluster (including the lower set).

It computes per-cluster thresholds (`alpha` and `beta`) to distinguish core vs. fringe members, then updates cluster centroids by mixing core and fringe means.
