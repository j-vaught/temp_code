from __future__ import annotations
"""
Histogram & stats per (range, gain).

Usable both as a **CLI script** and as an **importable helper**:

```python
from gain_statistics_helper import compute_histograms
compute_histograms("/path/to/output.nc", "results_dir")
```
"""
import argparse
import json
import pathlib
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np
from netCDF4 import Dataset
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm


class SNRMethod(str, Enum):
    STANDARD = "standard"
    FAST = "fast"
    SLOW = "slow"


def _worker(
    args: Tuple[
        str,
        int,
        int,
        np.ndarray,
        np.ndarray,
        int,
        int,
        SNRMethod,
    ]
) -> Tuple[int, int, np.ndarray, Dict[str, float]]:
    nc_path, rng_val, g_val, idx_rg, hist_bins, low_pctl, high_pctl, method = args
    # Load the raw echo data for this range/gain group as a 3D block
    with Dataset(nc_path, "r") as ds:
        raw3d = ds.variables["echo"][idx_rg, :, :].astype(np.float64)

    # Flatten for histogram & basic stats, dropping NaNs
    data = np.ma.masked_invalid(raw3d).compressed()

    # ── TRIM OUT 0’s and saturation max’s for baseline p_low ─────────────────
    if data.size:
        max_val = data.max()
        mask = (data != 0) & (data != max_val)
        trimmed = data[mask]
    else:
        trimmed = data

    if data.size == 0:
        empty = np.zeros_like(hist_bins[:-1], dtype=np.int64)
        stats = {"p_low": None, "p_high": None, "mean": None, "snr": None, "entropy": None, "coverage": 0}
        return rng_val, g_val, empty, stats

    # Histogram and percentiles
    hist, _    = np.histogram(data, bins=hist_bins)
    total      = hist.sum() or 1
    # use trimmed data for p_low if enough points, else full data
    if trimmed.size >= 10:
        p_low = np.percentile(trimmed, low_pctl)
    else:
        p_low = np.percentile(data,       low_pctl)
    p_high = np.percentile(data, high_pctl)
    mean_val = data.mean()


    # comnneted out all te other SNR calcualtion method. only Slow is implemented. Standard(bottom 5%) and Fast(cell v cell comparison) are commented out

    # Compute noise_sigma & SNR according to chosen method
    # if method == SNRMethod.STANDARD:
    #     noise_sigma = max(p_low, 1e-9)
    #     snr_val = mean_val / noise_sigma

    # elif method == SNRMethod.FAST:
    #     # Differences along fast-time (axis=1: gates) within same pulse/azimuth
    #     d = np.diff(raw3d, axis=1).ravel()
    #     med_d = np.median(d)
    #     mad_d = np.median(np.abs(d - med_d))
    #     thr = 3 * 1.4826 * mad_d
    #     d_small = d[np.abs(d - med_d) < thr]
    #     if d_small.size >= 10:
    #         noise_sigma = d_small.std(ddof=1) / np.sqrt(2)
    #     else:
    #         noise_sigma = max(p_low, 1e-9)
    #     baseline = p_low
    #     snr_val = (mean_val - baseline) / max(noise_sigma, 1e-9)

    # else:  # SNRMethod.SLOW
        # Differences along slow-time (axis=0: pulses) for each gate/azimuth
    d = np.diff(raw3d, axis=0).ravel()
    med_d = np.median(d)
    mad_d = np.median(np.abs(d - med_d))
    thr = 3 * 1.4826 * mad_d
    d_small = d[np.abs(d - med_d) < thr]
    if d_small.size >= 10:
        noise_sigma = d_small.std(ddof=1) / np.sqrt(2)
    else:
        noise_sigma = max(p_low, 1e-9)
    baseline = p_low
    snr_val = (mean_val - baseline) / max(noise_sigma, 1e-9)

    # Assemble stats
    stats = {
        "p_low": float(p_low),
        "p_high": float(p_high),
        "mean": float(mean_val),
        "snr": float(snr_val),
        "entropy": float(-np.sum((hist / total) * np.log2((hist / total) + 1e-12))),
        "coverage": int((hist > 0).sum()),
    }
    return rng_val, g_val, hist.astype(np.int64), stats


def compute_histograms(
    nc_file: str | pathlib.Path,
    output_dir: str | pathlib.Path = "analysis_out",
    hist_bins: np.ndarray | None = None,
    low_pctl: int = 5,
    high_pctl: int = 99,
    n_workers: int | None = None,
    snr_method: SNRMethod = SNRMethod.STANDARD,
) -> pathlib.Path:
    """
    Process *nc_file* and write .npy histograms plus JSON summary into *output_dir*.

    Returns
    -------
    pathlib.Path
        Directory where outputs are written.
    """
    nc_file = str(nc_file)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_hist = output_dir / "histograms"
    out_hist.mkdir(exist_ok=True)

    hist_bins = np.asarray(hist_bins) if hist_bins is not None else np.arange(0, 253)

    # Gather (range, gain) indices
    with Dataset(nc_file, "r") as ds:
        range_vals = ds.variables["range"][:].astype(int)
        gain_vals = ds.variables["gain"][:].astype(int)

    tasks: List[Tuple[str, int, int, np.ndarray, np.ndarray, int, int, SNRMethod]] = []
    for r in np.unique(range_vals):
        mask_r = range_vals == r
        for g in np.unique(gain_vals[mask_r]):
            idx_rg = np.where(mask_r & (gain_vals == g))[0]
            tasks.append((
                nc_file,
                r,
                g,
                idx_rg,
                hist_bins,
                low_pctl,
                high_pctl,
                snr_method,
            ))

    all_stats: Dict[Tuple[int, int], Dict[str, float]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for rng_val, g_val, hist, stats in tqdm(
            ex.map(_worker, tasks),
            total=len(tasks),
            desc="Groups",
        ):
            np.save(out_hist / f"hist_r{rng_val}_g{g_val}.npy", hist)
            all_stats[(rng_val, g_val)] = stats

    summary = {f"r{r}_g{g}": s for (r, g), s in all_stats.items()}
    (output_dir / "summary_stats.json").write_text(json.dumps(summary, indent=2))
    return output_dir


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-(range,gain) histograms.")
    p.add_argument("nc_file", help="Input NetCDF file")
    p.add_argument(
        "-o",
        "--output-dir",
        default="analysis_out",
        help="Output directory",
    )
    p.add_argument(
        "--low-pctl",
        type=int,
        default=5,
        help="Low percentile for noise baseline",
    )
    p.add_argument(
        "--high-pctl",
        type=int,
        default=99,
        help="High percentile for stats",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Process pool size (default=os.cpu_count())",
    )
    p.add_argument(
        "--snr-method",
        choices=[m.value for m in SNRMethod],
        default=SNRMethod.STANDARD.value,
        help=(
            "Which SNR estimator to use: "
            "standard=mean/p_low, fast=fast-time diffs, slow=slow-time diffs"
        ),
    )
    return p.parse_args()


def main() -> None:
    ns = _parse_cli()
    compute_histograms(
        nc_file=ns.nc_file,
        output_dir=ns.output_dir,
        low_pctl=ns.low_pctl,
        high_pctl=ns.high_pctl,
        n_workers=ns.workers,
        # snr_method=SNRMethod(ns.snr_method),
        snr_method=SNRMethod.SLOW,
    )
    print(f"✓ Completed processing – outputs in {ns.output_dir}")


if __name__ == "__main__":
    main()
