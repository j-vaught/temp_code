# Radar Data Processing Toolkit

This repository contains a mixture of Python and Rust utilities for working with raw marine radar data.  The code converts CSV sweeps into NetCDF files, produces visualizations, performs gain analysis and clutter removal, and includes several helper command–line tools.

## Repository layout

```
PYTHON/  - stand‑alone Python scripts
RUST/    - Rust command line applications
```

Below is an overview of each component.

## Python scripts (`PYTHON/`)

| File / Directory | Purpose |
|------------------|---------|
| `csv_netcdf.py` | Convert a directory of `good/*.csv` sweeps into a compressed NetCDF file. Uses multiprocessing and NetCDF4 with optional parallel I/O. |
| `csv_to_png.py` | Render a single CSV sweep as a polar PNG image using Pillow. |
| `netcdf_png.py` | Read the NetCDF output and generate per‑frame PNGs using NumPy/Numba for polar‑to‑Cartesian mapping. |
| `png_mp4.py` | Compose overlay PNG frames with a satellite imagery background and encode an MP4 video with MoviePy. |
| `gain_statistics_helper.py` | Compute per‑(range,gain) histograms from the NetCDF file and output summary statistics. |
| `gain_analysis_plots_summary.py` | Post‑process the histograms, select the best gains, and generate CSV/plot summaries. |
| `test_ppt_img.py` | Miscellaneous 3‑D plotting and animation example used for presentations. |
| `CLUTTER/` | Scripts for clutter removal experiments including Noise2Void training/inference and HDR fusion. |
| `old_python/` | Collection of older scripts and MATLAB experiments kept for reference. |

All Python code expects NumPy, Pandas and other scientific packages.  Many scripts contain absolute paths that should be edited before running.

## Rust crates (`RUST/`)

| Crate | Description |
|-------|-------------|
| `radar_to_netcdf` | High‑performance converter from CSV sweeps to a CF‑compliant NetCDF file. Uses Rayon for parallelism and bundles the `netcdf-c` library (see `netcdf-c-4.9.2/`). |
| `angle_correction` | Utility that scans CSV files and fixes duplicated angle values when interpolation was disabled. |
| `check_csv_inc` | Classify CSVs as **bad**, **gain_change** or **good** by checking angle monotonicity and gain consistency. Can optionally move files into subfolders. |
| `csv_dim_reporter` | Library with helpers for parsing sweep metadata. The `main.rs` stub shows how to build a NetCDF generator. |

Each crate is a standalone Cargo project; run with `cargo run --release` from its directory.

## NetCDF library

`RUST/radar_to_netcdf/netcdf-c-4.9.2/` contains a vendored copy of Unidata’s NetCDF‑C library used when building the Rust converter.  See its `README.md` for details.

---
This README is only a high‑level guide.  Inspect individual source files for implementation details and adjust paths in the scripts before execution.
