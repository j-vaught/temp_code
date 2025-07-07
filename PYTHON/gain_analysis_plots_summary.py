from __future__ import annotations
"""
Post-process saved histograms, pick best gains *and* plot summary curves.
If histogram files are missing, they are generated automatically by calling
`gain_statistics_helper.compute_histograms()` – no more hard‑coded paths!
"""
import argparse
import json
import math
import pathlib
import re
import warnings
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from sklearn.cluster import KMeans
except ImportError:
    KMeans = None
    warnings.warn("scikit-learn not installed – K-means selection skipped.")

from gain_statistics_helper import compute_histograms, SNRMethod

# ─────────────── CLI & defaults ────────────────
DEFAULT_N_BEST = 5
DEFAULT_K_RANGE = [3, 4, 5]
DEFAULT_LOW_Q, DEFAULT_HI_Q = 0.05, 0.99
DEFAULT_FMT = "png"

def _ensure_histograms(
    base_dir: pathlib.Path,
    nc_file: str | None,
    snr_method: SNRMethod,
) -> None:
    hist_dir = base_dir / "histograms"
    if hist_dir.exists() and any(hist_dir.glob("hist_r*_g*.npy")):
        return
    if not nc_file:
        sys.exit("Histogram directory missing and --nc-file not supplied.")
    print("⏳ Histogram files missing – generating them first…")
    compute_histograms(
        nc_file=nc_file,
        output_dir=base_dir,
        snr_method=snr_method,
    )


def main() -> None:

    # ——— VS Code “Run” settings (no CLI needed) ———
    # Just edit these constants and hit Run
    BASE_DIR    = "output_stats_6"
    NC_FILE     = "/Users/jacobvaught/Downloads/Research_radar_DATA/data/data_6/output.nc"      # or None if you already have histograms
    SNR_METHOD  = SNRMethod.STANDARD
    N_BEST      = DEFAULT_N_BEST
    K_RANGE     = DEFAULT_K_RANGE
    LOW_Q, HI_Q = DEFAULT_LOW_Q, DEFAULT_HI_Q
    PLOT_FMT    = DEFAULT_FMT
    BIN_EDGES   = np.arange(0, 253)

    base_dir = pathlib.Path(BASE_DIR)
    hist_dir = base_dir / "histograms"
    _ensure_histograms(base_dir, NC_FILE, SNR_METHOD)

    # settings
    n_best   = N_BEST
    k_range  = K_RANGE
    low_q, hi_q = LOW_Q, HI_Q
    plot_fmt = PLOT_FMT
    bin_edges = BIN_EDGES

    # 1. Load histograms → metrics
    records: List[dict] = []
    re_fname = re.compile(r"hist_r(?P<range>\d+)_g(?P<gain>\d+)\.npy")
    for file in sorted(hist_dir.glob("hist_r*_g*.npy")):
        m = re_fname.match(file.name)
        if not m:
            continue
        r, g = int(m["range"]), int(m["gain"])
        hist = np.load(file)
        total = hist.sum()
        if total == 0:
            continue

        # basic stats
        cdf = np.cumsum(hist)
        idx5 = np.searchsorted(cdf, low_q * total)
        idx99 = np.searchsorted(cdf, hi_q * total)
        coverage = int((hist > 0).sum())
        probs = hist / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

        bin_vals = bin_edges[:-1] + 0.5
        mean_val = float((bin_vals * hist).sum() / total)
        mask_low = np.arange(len(hist)) <= idx5
        noise_sigma = float(
            math.sqrt(
                ((hist[mask_low] * (bin_vals[mask_low] - mean_val) ** 2).sum())
                / max(hist[mask_low].sum(), 1)
            )
        )
        snr_lin = mean_val / noise_sigma if noise_sigma > 0 else math.nan

        records.append(
            dict(
                range=r,
                gain=g,
                p_low=int(idx5),
                p_high=int(idx99),
                coverage=coverage,
                entropy=entropy,
                snr=snr_lin,
            )
        )

    df = pd.DataFrame(records)
    csv_path = base_dir / "hist_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"✓ Wrote per‑gain metrics → {csv_path}")

    # 2. Gain selection algorithms
    def _greedy_set_cover(subsets: Dict[int, np.ndarray], n_bins: int, n_best: int) -> List[int]:
        covered = np.zeros(n_bins, bool)
        picks: List[int] = []
        while len(picks) < n_best:
            best_gain, best_add = None, 0
            for g, bins in subsets.items():
                new = (~covered) & bins
                if new.sum() > best_add:
                    best_add, best_gain = new.sum(), g
            if best_gain is None or best_add == 0:
                break
            picks.append(best_gain)
            covered |= subsets[best_gain]
        return picks

    selections: Dict[int, Dict[str, List[int]]] = {}
    for rng, grp in df.groupby("range"):
        masks = {
            int(row.gain): (np.load(hist_dir / f"hist_r{rng}_g{int(row.gain)}.npy") > 0)
            for _, row in grp.iterrows()
        }
        picks_cov = _greedy_set_cover(masks, len(bin_edges) - 1, n_best)

        union = np.zeros(len(bin_edges) - 1, int)
        picks_ent: List[int] = []
        avail = set(masks)
        for _ in range(n_best):
            best_gain, best_ent = None, -1
            for g in avail:
                cand = union + np.load(hist_dir / f"hist_r{rng}_g{g}.npy")
                p = cand / cand.sum()
                ent = -np.sum(p * np.log2(p + 1e-12))
                if ent > best_ent:
                    best_ent, best_gain = ent, g
            if best_gain is None:
                break
            picks_ent.append(best_gain)
            union += np.load(hist_dir / f"hist_r{rng}_g{best_gain}.npy")
            avail.remove(best_gain)

        picks_km: Dict[int, List[int]] = {}
        if KMeans and not grp[["p_low", "p_high", "snr"]].isna().any().any():
            feats = grp[["p_low", "p_high", "snr"]].to_numpy()
            for k in DEFAULT_K_RANGE:
                if k > len(feats):
                    continue
                km = KMeans(n_clusters=k, init="k-means++", random_state=42).fit(feats)
                labels, centers = km.labels_, km.cluster_centers_
                picks: List[int] = []
                for c in range(k):
                    idx = np.where(labels == c)[0]
                    d = np.linalg.norm(feats[idx] - centers[c], axis=1)
                    picks.append(int(grp.iloc[idx[d.argmin()]].gain))
                picks_km[k] = picks

        selections[rng] = dict(greedy=picks_cov, entropy=picks_ent, kmeans=picks_km)

    with open(base_dir / "hist_selected.json", "w") as f:
        json.dump(selections, f, indent=2)
    print(f"✓ Gain selections → {base_dir/'hist_selected.json'}")

    # 3. Plots
    print("📈  Generating summary plots…")
    ranges = sorted(df["range"].unique())

    def _save_lineplot(y_col: str, ylabel: str, fname: str) -> None:
        plt.figure()
        for r in ranges:
            sub = df[df["range"] == r].sort_values("gain")
            plt.plot(sub["gain"], sub[y_col], marker="o", label=f"range {r}")
        plt.xlabel("Gain")
        plt.ylabel(ylabel)
        plt.title(f"{ylabel} vs Gain")
        plt.legend()
        plt.tight_layout()
        plt.savefig(base_dir / f"plot_{fname}.{plot_fmt}")
        plt.close()

    _save_lineplot("coverage", "Coverage (non‑zero bins)", "coverage")
    _save_lineplot("entropy", "Shannon entropy (bits)", "entropy")
    _save_lineplot("snr", "Approx. SNR (linear)", "snr")

    print("✓  Plots saved to:", base_dir)


if __name__ == "__main__":
    main()