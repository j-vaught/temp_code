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
| `CLUTTER/` | Scripts for clutter-removal experiments (see [CLUTTER details](#clutter-experiments-pythonclutter)). |
| `old_python/` | Archive of older scripts and MATLAB experiments (see [old_python details](#archived-scripts-pythonold_python)). |

### CLUTTER experiments (`PYTHON/CLUTTER`)

- `MAE_test.py` – PyTorch Masked Autoencoder training demo with TensorBoard logging.
- `N2V_Training.py` – Patch-based Noise2Void training pipeline.
- `N2V_Inference.py` – Apply a trained Noise2Void model to a single PNG frame.
- `clutter_clearance_data9.py` – Offline clutter-removal pipeline with MTI subtraction, wavelets, RPCA, multi-frame denoise and HDR fusion options.
- `clutter_clearance_data9_v2.py` – Simplified variant supporting multi-look averaging, Lee filter, BM3D/Noise2Void and optional HDR fusion.

### Archived scripts (`PYTHON/old_python`)

- `Lake_map_generator_SC.py` – Download South Carolina lakes and cities with OSMnx and plot them.
- `Map_url_to_lat-long.py` – Resolve Google Maps share URLs to latitude/longitude coordinates.
- `manual_overlay.py` – Interactive polygon mask tool with feathered ramps.
- `MiDAS_depth.py` – Experiment with MiDaS depth estimation and orthorectification.
- `segmentation.py` – Run an ONNX segmentation model on tiled image regions.
- `testMAE.py` / `test_MAE_model.py` – Prototype convolutional MAE training and inference on NetCDF radar sweeps.
- `multi-gain_distribution.py` – Radar simulator with gain sweeps and histogram plots.
- `test_denoise.py` – Multi-gain simulation plus Ridge-regression clutter removal.
- `radar_simulations/` – Collection of small simulation modules and CFAR examples.
- `TXT.NC_script.py` – Convert a ZIP of text sweeps into a NetCDF dataset.
- `test_png_data.py` – Extract EXIF metadata from sample images.
- `plot_radar_sweep.m` – MATLAB routine for visualizing radar data.
- `.xlsx`/`.csv` files – Mapping datasets used by other scripts.

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
