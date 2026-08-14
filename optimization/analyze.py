#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 The SCPP developers. See LICENSE for details.

"""Turn raw measurements into the study's results, tables and figures.

Reads every ``optimization/benchmarks/raw/*.jsonl`` produced by ``sweep.py``
and the correctness JSON produced by ``compare.py``, consolidates them into
``results.csv`` / ``results.json``, and regenerates every table and figure from
that consolidated file — so the artefacts can never drift from the data.

Usage
-----
    python optimization/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "benchmarks" / "raw"
BENCH = HERE / "benchmarks"
FIGS = HERE / "figures"
REPORTS = HERE / "reports"
CORRECT = HERE / "correctness"

# Which raw file is which build. Files not listed are exploratory (the initial
# survey and the profiling runs) and are kept in results.csv but excluded from
# the paired original-vs-optimized comparison.
BUILDS = {
    "sdbg_original": ("SoftDBSCANGM", "original"),
    "sdbg_optimized": ("SoftDBSCANGM", "optimized"),
    "mbmm_original": ("MBMM", "original"),
    "mbmm_optimized": ("MBMM", "optimized"),
    "kfcm_original": ("KFCM", "original"),
    "kfcm_optimized": ("KFCM", "optimized"),
    "kfccl_original": ("KFCCL", "original"),
    "kfccl_optimized": ("KFCCL", "optimized"),
    "kmart_original": ("KMART", "original"),
    "kmart_optimized": ("KMART", "optimized"),
    "smoke": (None, "survey"),
    "profile_top": (None, "profile"),
    "profile_slow": (None, "profile"),
}

FAMILY = {
    "SoftDBSCANGM": "density / mixture",
    "MBMM": "mixture",
    "KFCM": "kernel",
    "KFCCL": "kernel",
    "KMART": "document",
}


# --------------------------------------------------------------------------
# Consolidation
# --------------------------------------------------------------------------


def load_rows() -> list[dict]:
    rows = []
    for path in sorted(RAW.glob("*.jsonl")):
        stem = path.stem
        _, build = BUILDS.get(stem, (None, stem))
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            shape = rec.get("membership_shape") or [None, None]
            rows.append(
                {
                    "algorithm": rec.get("case"),
                    "family": FAMILY.get(rec.get("case"), rec.get("family", "")),
                    "implementation": build,
                    "source_file": stem,
                    "dataset": rec.get("modality", ""),
                    "n": rec.get("n"),
                    "d": rec.get("d"),
                    "k": rec.get("k"),
                    "k_fitted": shape[1],
                    "status": rec.get("status"),
                    "fit_time_ms": rec.get("fit_time_ms"),
                    "fit_time_mean_ms": rec.get("fit_time_mean_ms"),
                    "repeats": len(rec.get("times_s") or []),
                    "peak_traced_mb": rec.get("peak_traced_mb"),
                    "timeout_s": rec.get("timeout_s"),
                }
            )
    return rows


def attach_correctness(rows: list[dict]) -> dict:
    """Fold the correctness comparison into a per-algorithm summary."""
    summary: dict[str, dict] = {}
    for path in sorted(CORRECT.glob("*.json")):
        records = json.loads(path.read_text())
        if not records:
            continue
        alg = records[0]["case"]
        mem = [
            r["memberships"]
            for r in records
            if r["memberships"].get("max_abs") is not None
        ]
        summary[alg] = {
            "n_comparisons": len(records),
            "max_membership_abs_error": max((m["max_abs"] for m in mem), default=None),
            "mean_membership_abs_error": (
                float(np.mean([m["mean_abs"] for m in mem])) if mem else None
            ),
            "max_membership_rel_error": max((m["max_rel"] for m in mem), default=None),
            "min_label_agreement": min(
                (r.get("label_agreement", 1.0) for r in records), default=None
            ),
            "cluster_count_matches": all(
                r["n_clusters_ref"] == r["n_clusters_opt"] for r in records
            ),
            "invariants_hold": all(
                all(
                    v
                    for kk, v in r.get("invariants", {}).items()
                    if isinstance(v, bool) and kk != "rows_sum_to_one"
                )
                for r in records
            ),
        }
    return summary


def paired(rows: list[dict]) -> dict[str, dict[int, dict]]:
    """{algorithm: {n: {'original': row, 'optimized': row}}}"""
    out: dict[str, dict[int, dict]] = {}
    for r in rows:
        if r["implementation"] not in ("original", "optimized"):
            continue
        out.setdefault(r["algorithm"], {}).setdefault(r["n"], {})[
            r["implementation"]
        ] = r
    return out


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------


def _speedup(o, p):
    if not o or not p:
        return None
    return o / p


def build_tables(rows, pairs, correctness) -> dict[str, str]:
    tables: dict[str, str] = {}

    # ---- Table 2: runtime comparison -------------------------------------
    lines = [
        "| Algorithm | n | Original (ms) | Optimized (ms) | Speedup | Reduction |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    tex2 = []
    for alg, by_n in sorted(pairs.items()):
        for n in sorted(by_n):
            o = by_n[n].get("original")
            p = by_n[n].get("optimized")
            if not o or not p:
                continue
            if o["status"] == "timeout":
                ot = f"timeout (> {o['timeout_s'] * 1000:,.0f})"
                sp = "> " + f"{o['timeout_s'] * 1000 / p['fit_time_ms']:,.0f}x"
                red = "> 99.9%"
                ot_tex = f"timeout ($>$ {o['timeout_s'] * 1000:,.0f})"
            elif o["status"] == "ok" and p["status"] == "ok":
                ot = f"{o['fit_time_ms']:,.1f}"
                s = _speedup(o["fit_time_ms"], p["fit_time_ms"])
                sp = f"{s:,.1f}x"
                red = f"{100 * (1 - p['fit_time_ms'] / o['fit_time_ms']):.1f}%"
                ot_tex = ot
            else:
                continue
            pt = f"{p['fit_time_ms']:,.1f}"
            lines.append(f"| {alg} | {n:,} | {ot} | {pt} | {sp} | {red} |")
            # Escape for LaTeX explicitly: a bare '%' opens a comment and would
            # swallow the rest of the row. Only the speedup's trailing 'x'
            # becomes \times -- a blanket replace would also hit algorithm names.
            sp_tex = sp.replace("x", r"$\times$").replace(">", "$>$")
            red_tex = red.replace("%", r"\%").replace(">", "$>$")
            tex2.append(f"{alg} & {n:,} & {ot_tex} & {pt} & {sp_tex} & {red_tex} \\\\")
    tables["table2_runtime.md"] = "\n".join(lines) + "\n"
    tables["table2_runtime.tex"] = _latex(
        "Runtime comparison, original versus optimized implementations.",
        "tab:runtime",
        "lrrrrr",
        ["Algorithm", "$n$", "Original (ms)", "Optimized (ms)", "Speedup", "Reduction"],
        tex2,
    )

    # ---- Table 3: memory --------------------------------------------------
    lines = [
        "| Algorithm | n | Original (MB) | Optimized (MB) | Change |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    tex3 = []
    for alg, by_n in sorted(pairs.items()):
        for n in sorted(by_n):
            o, p = by_n[n].get("original"), by_n[n].get("optimized")
            if not o or not p or o["status"] != "ok" or p["status"] != "ok":
                continue
            om, pm = o["peak_traced_mb"], p["peak_traced_mb"]
            if om is None or pm is None:
                continue
            change = 100 * (om - pm) / om if om else float("nan")
            lines.append(f"| {alg} | {n:,} | {om:.2f} | {pm:.2f} | {change:+.1f}% |")
            tex3.append(f"{alg} & {n:,} & {om:.2f} & {pm:.2f} & {change:+.1f}\\% \\\\")
    tables["table3_memory.md"] = "\n".join(lines) + "\n"
    tables["table3_memory.tex"] = _latex(
        "Peak traced Python allocation. A negative change denotes higher usage "
        "by the optimized implementation.",
        "tab:memory",
        "lrrrr",
        ["Algorithm", "$n$", "Original (MB)", "Optimized (MB)", "Change"],
        tex3,
    )

    # ---- Table 4: correctness --------------------------------------------
    lines = [
        "| Algorithm | Comparisons | Max membership error | Max relative error | "
        "Label agreement | Cluster count | Status |",
        "| --- | ---: | ---: | ---: | ---: | :-: | --- |",
    ]
    tex4 = []
    for alg, c in sorted(correctness.items()):
        status = (
            "EQUIVALENT"
            if (c["max_membership_abs_error"] or 0) < 1e-9
            and (c["min_label_agreement"] or 0) == 1.0
            and c["cluster_count_matches"]
            else "REVIEW"
        )
        lines.append(
            f"| {alg} | {c['n_comparisons']} | {c['max_membership_abs_error']:.3e} | "
            f"{c['max_membership_rel_error']:.3e} | "
            f"{c['min_label_agreement']:.4f} | "
            f"{'yes' if c['cluster_count_matches'] else 'no'} | {status} |"
        )
        tex4.append(
            f"{alg} & {c['n_comparisons']} & "
            f"{c['max_membership_abs_error']:.2e} & "
            f"{c['max_membership_rel_error']:.2e} & "
            f"{c['min_label_agreement']:.4f} & "
            f"{'yes' if c['cluster_count_matches'] else 'no'} & {status} \\\\"
        )
    tables["table4_correctness.md"] = "\n".join(lines) + "\n"
    tables["table4_correctness.tex"] = _latex(
        "Correctness of the optimized implementations against the preserved "
        "reference implementations.",
        "tab:correctness",
        "lrrrrcl",
        [
            "Algorithm",
            "Comparisons",
            "Max abs.\\ error",
            "Max rel.\\ error",
            "Label agr.",
            "$k$ match",
            "Status",
        ],
        tex4,
    )

    # ---- Table 5: scalability --------------------------------------------
    lines = [
        "| Algorithm | n | Original | Optimized | Speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for alg, by_n in sorted(pairs.items()):
        for n in sorted(by_n):
            o, p = by_n[n].get("original"), by_n[n].get("optimized")
            if not p:
                continue
            ot = (
                f"{o['fit_time_ms']:,.1f} ms"
                if o and o["status"] == "ok"
                else ("timeout" if o else "n/a")
            )
            sp = (
                f"{_speedup(o['fit_time_ms'], p['fit_time_ms']):,.1f}x"
                if o and o["status"] == "ok"
                else "—"
            )
            lines.append(
                f"| {alg} | {n:,} | {ot} | {p['fit_time_ms']:,.1f} ms | {sp} |"
            )
    tables["table5_scalability.md"] = "\n".join(lines) + "\n"

    return tables


def _latex(caption, label, colspec, headers, body_rows) -> str:
    head = " & ".join(headers) + r" \\"
    return (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
        f"{head}\n\\midrule\n" + "\n".join(body_rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
        }
    )
    return plt


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}")
    print(f"  figure: {name}.png / .pdf")


ORIG_C, OPT_C = "#B0562A", "#2A6FB0"


def figures(rows, pairs, correctness):
    plt = setup_mpl()

    # --- Figure 1: speedup by algorithm and size --------------------------
    labels, values = [], []
    for alg, by_n in sorted(pairs.items()):
        for n in sorted(by_n):
            o, p = by_n[n].get("original"), by_n[n].get("optimized")
            if o and p and o["status"] == "ok" and p["status"] == "ok":
                labels.append(f"{alg}\nn={n:,}")
                values.append(o["fit_time_ms"] / p["fit_time_ms"])
    order = np.argsort(values)
    fig, ax = plt.subplots(figsize=(6.0, 0.42 * len(labels) + 1.2))
    ax.barh(
        [labels[i] for i in order], [values[i] for i in order], color=OPT_C, height=0.62
    )
    for y, i in enumerate(order):
        ax.text(values[i] * 1.05, y, f"{values[i]:,.0f}x", va="center", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Speedup (original / optimized fit time), log scale")
    ax.set_title("Runtime speedup of optimized implementations")
    ax.set_xlim(right=max(values) * 3)
    save(fig, "fig1_speedup")
    plt.close(fig)

    # --- Figure 2 & 3: runtime and memory, before vs after ----------------
    for figname, key, ylabel, title, logy in (
        (
            "fig2_runtime_comparison",
            "fit_time_ms",
            "Fit time (ms)",
            "Runtime before and after optimization",
            True,
        ),
        (
            "fig3_memory_comparison",
            "peak_traced_mb",
            "Peak traced allocation (MB)",
            "Memory before and after optimization",
            True,
        ),
    ):
        labels, ov, pv = [], [], []
        for alg, by_n in sorted(pairs.items()):
            for n in sorted(by_n):
                o, p = by_n[n].get("original"), by_n[n].get("optimized")
                if not (o and p and o["status"] == "ok" and p["status"] == "ok"):
                    continue
                if o.get(key) is None or p.get(key) is None:
                    continue
                labels.append(f"{alg}\nn={n:,}")
                ov.append(o[key])
                pv.append(p[key])
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(1.15 * len(labels) + 1.6, 3.2))
        ax.bar(x - 0.2, ov, 0.4, label="original", color=ORIG_C)
        ax.bar(x + 0.2, pv, 0.4, label="optimized", color=OPT_C)
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title)
        ax.legend(frameon=False)
        save(fig, figname)
        plt.close(fig)

    # --- Figure 4: scalability -------------------------------------------
    # Wrapped into a grid rather than one long row: a single row of panels
    # exceeds a printed text column once more than three algorithms are paired.
    algs = sorted(pairs)
    ncols = min(3, len(algs))
    nrows = -(-len(algs) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.6 * ncols, 3.1 * nrows), squeeze=False
    )
    flat_axes = [ax for row in axes for ax in row]
    for extra in flat_axes[len(algs) :]:
        extra.axis("off")
    for ax, alg in zip(flat_axes, algs):
        by_n = pairs[alg]
        ns = sorted(by_n)
        for build, colour, marker in (
            ("original", ORIG_C, "o"),
            ("optimized", OPT_C, "s"),
        ):
            xs = [n for n in ns if by_n[n].get(build, {}).get("status") == "ok"]
            ys = [by_n[n][build]["fit_time_ms"] for n in xs]
            if xs:
                ax.plot(xs, ys, marker=marker, color=colour, label=build, markersize=4)
        # mark timeouts
        touts = [
            n for n in ns if by_n[n].get("original", {}).get("status") == "timeout"
        ]
        for n in touts:
            lim = by_n[n]["original"]["timeout_s"] * 1000
            ax.plot([n], [lim], marker="x", color=ORIG_C, markersize=8)
            ax.annotate(
                "timeout",
                (n, lim),
                textcoords="offset points",
                xytext=(-6, 6),
                fontsize=7,
                color=ORIG_C,
                ha="right",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Label only the sizes actually measured; the default log locator
        # crowds minor ticks into an unreadable overlap at these ranges.
        ax.set_xticks(ns, [f"{n:,}" for n in ns], fontsize=8)
        ax.set_xticks([], minor=True)
        ax.set_xlabel("samples $n$")
        ax.set_ylabel("fit time (ms)")
        ax.set_title(alg)
        ax.legend(frameon=False)
    fig.suptitle("Scalability in the sample count", y=1.00)
    # Needed once the panels wrap: without it the first row's x-labels collide
    # with the second row's titles.
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "fig4_scalability")
    plt.close(fig)

    # --- Figure 5: accuracy preservation ---------------------------------
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    names, errs = [], []
    for alg, c in sorted(correctness.items()):
        names.append(alg)
        errs.append(max(c["max_membership_abs_error"], 1e-34))
    ax.barh(names, errs, color=OPT_C, height=0.5)
    ax.axvline(np.finfo(float).eps, color="grey", ls="--", lw=1)
    ax.annotate(
        "machine epsilon",
        (np.finfo(float).eps, -0.42),
        fontsize=7,
        color="grey",
        rotation=90,
        va="bottom",
        ha="right",
    )
    ax.set_xscale("log")
    ax.set_xlabel(r"max $|U_{\mathrm{original}} - U_{\mathrm{optimized}}|$, log scale")
    ax.set_title("Membership agreement with the reference")
    for y, e in enumerate(errs):
        ax.text(e * 1.6, y, f"{e:.1e}", va="center", fontsize=7)
    ax.set_xlim(right=max(errs) * 400)
    save(fig, "fig5_accuracy_preservation")
    plt.close(fig)

    # --- Figure 6: baseline landscape across the surveyed library ---------
    survey = [
        r
        for r in rows
        if r["implementation"] == "survey" and r["status"] == "ok" and r["fit_time_ms"]
    ]
    survey.sort(key=lambda r: r["fit_time_ms"])
    fig, ax = plt.subplots(figsize=(6.0, 0.26 * len(survey) + 1.2))
    colours = [OPT_C if r["algorithm"] in pairs else "#9aa6b2" for r in survey]
    ax.barh(
        [r["algorithm"] for r in survey],
        [r["fit_time_ms"] for r in survey],
        color=colours,
        height=0.68,
    )
    ax.set_xscale("log")
    ax.set_xlabel("baseline fit time at $n=200$ (ms), log scale")
    ax.set_title("Baseline runtime landscape (pre-optimization survey)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=OPT_C),
        plt.Rectangle((0, 0), 1, 1, color="#9aa6b2"),
    ]
    ax.legend(
        handles,
        ["optimized in this study", "surveyed only"],
        frameon=False,
        loc="lower right",
    )
    save(fig, "fig6_baseline_landscape")
    plt.close(fig)


# --------------------------------------------------------------------------


def main() -> int:
    rows = load_rows()
    correctness = attach_correctness(rows)
    pairs = paired(rows)

    BENCH.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    import csv

    with (BENCH / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (BENCH / "results.json").write_text(
        json.dumps({"measurements": rows, "correctness": correctness}, indent=2)
    )
    print(f"wrote {BENCH/'results.csv'} ({len(rows)} rows)")
    print(f"wrote {BENCH/'results.json'}")

    tables = build_tables(rows, pairs, correctness)
    tdir = REPORTS / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    for name, content in tables.items():
        (tdir / name).write_text(content)
        print(f"  table: {name}")

    figures(rows, pairs, correctness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
