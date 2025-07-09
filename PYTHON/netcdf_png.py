"""
=========================================================================================
Batch Radar Data → Image Renderer (Parallelized, Mac-optimized)
=========================================================================================

This script batch-renders frames from a polar radar dataset (NetCDF format) into PNG images.
It is designed for speed via parallel processing (using all available CPU cores), efficient
memory use, and optional Apple-GPU (MPS) acceleration for logarithmic transforms on MacOS.

-----------------------------------------------------------------------------------------
WHAT IT DOES:
- Loads a NetCDF radar dataset containing polar "echo" sweeps and angle/time metadata.
- For each frame (time-step):
    1. Extracts valid angular sweeps and applies masking.
    2. Transforms polar data to Cartesian pixel space.
    3. Applies log transform, normalization, and maps values to a color palette (colormap).
    4. Renders to an RGBA (with transparency) PNG image, one per time-step.
    5. Filenames are info-rich: include time, range, scale, and gain per frame.
- Runs in parallel (across CPU cores) with non-blocking I/O for fast PNG encoding.
- Output images are written to an output directory, suitable for making movies/animations.

-----------------------------------------------------------------------------------------
INPUTS:
- FILE:      Path to input NetCDF file (see 'USER CONFIG' below).
- ECHO_VAR:  Variable name in the NetCDF file for radar reflectivity/echo.
- ANGLE_VAR: Variable name in the NetCDF file for angles.
- Other metadata variables (e.g., 'range', 'scale', 'gain', 'time') must also be present.

-----------------------------------------------------------------------------------------
OUTPUTS:
- Directory of PNG images, one per radar frame, named like:
    frame_00001_tYYYYMMDDTHHMMSS_rcNN_ucSS_gG.png
    (Where time, range, scale, and gain are frame-specific.)

-----------------------------------------------------------------------------------------
CUSTOMIZATION (see USER CONFIG section):
- FILE:        Input NetCDF file path.
- ECHO_VAR:    Name of radar/echo variable.
- ANGLE_VAR:   Name of angle variable.
- PULSE_STEP:  Downsampling factor (use 1 for all pulses, 2 for every other, etc).
- USE_MPS:     Enable Apple MPS (Metal Performance Shaders) acceleration for log1p.
- OUT_DIR:     Output directory for PNGs.
- MAX_WORKERS: Number of parallel processes (default: all CPU cores).

-----------------------------------------------------------------------------------------
DEPENDENCIES:
- numpy, netCDF4, xarray, matplotlib, Pillow, numba, torch, tqdm, pandas, scikit-learn, dask
  (see pip install line below)

-----------------------------------------------------------------------------------------
USAGE:
    # (Recommended environment setup, for new venvs or conda installs)
    pip install numpy netCDF4 tqdm mpi4py xarray Pillow moviepy requests pandas matplotlib scikit-learn numba torch dask

    # Run script directly:
    python this_script.py

    # Output PNGs will appear in OUT_DIR.

-----------------------------------------------------------------------------------------
NOTES:
- Output images are log-transformed, colormapped, and alpha-blended (transparent background).
- Code is portable, but some GPU acceleration is MacOS-only (torch + MPS).
- All parallel workers initialize their own data handles and I/O threads.

-----------------------------------------------------------------------------------------
"""


# >standard library imports>
import io, os, time, queue, threading
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# <third‑party imports>
import numpy as np
import xarray as xr
from matplotlib import cm
from PIL import Image
from numba import njit, prange
import datetime

# ───── USER CONFIG ────────────────────────────────────────────────
FILE        = "/Users/jacobvaught/Downloads/Research_radar_DATA/sample/test/output.nc"
ECHO_VAR    = "echo"
ANGLE_VAR   = "angle"
PULSE_STEP  = 1                 # keep every pulse, or thin by 2,4,…
USE_MPS     = True              # Apple‑GPU for log1p if available
OUT_DIR     = "frames_parallel_9g" # destination for PNGs
MAX_WORKERS = os.cpu_count()    # parallelism – leave at os.cpu_count()
# ──────────────────────────────────────────────────────────────────

@njit(parallel=True)
def remap_colormap(H, W, mask, angle_idx, gate_idx, col_idx, col_tab, out_img):
    """Vectorised polar‑to‑Cartesian colour mapping (unchanged)."""
    for i in prange(H):
        for j in range(W):
            if mask[i, j]:
                idx = col_idx[angle_idx[i, j], gate_idx[i, j]]
                out_img[i, j, 0] = col_tab[idx, 0]
                out_img[i, j, 1] = col_tab[idx, 1]
                out_img[i, j, 2] = col_tab[idx, 2]
                out_img[i, j, 3] = col_tab[idx, 3]
            else:
                out_img[i, j, 3] = 0  # transparent background

# ensure output directory exists
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ───── Worker initialisation (runs once per process) ──────────────

def init_worker(use_mps: bool):
    import torch  # local import: each worker checks MPS availability

    global DS, VALID, VMIN, VMAX
    global COL_TAB, MASK_MAP, ANGLE_IDX_MAP, GATE_IDX_MAP
    global MPS_OK, IO_Q
    global TIME_COORD, RANGE_CODE, SCALE_CODE, GAIN  # per‑frame metadata arrays

    DS = xr.open_dataset(FILE, chunks={"time": 1})

    # store metadata for filename generation
    TIME_COORD = DS["time"].values                 # (time,)
    RANGE_CODE = DS["range"].astype(int).values    # (time,)
    SCALE_CODE = DS["scale"].astype(int).values    # (time,)
    GAIN       = DS["gain"].astype(int).values     # (time,)

    # geometry mask derived from the first sweep (unchanged logic)
    angles0 = DS[ANGLE_VAR].isel(time=0).values
    VALID   = ~np.isnan(angles0)[::PULSE_STEP]
    θ       = np.deg2rad(angles0[::PULSE_STEP][VALID])
    N_theta = θ.size

    # dynamic vmin/vmax over the whole dataset (sample 10 %)
    n_frames = DS.dims["time"]
    samp_idx = slice(0, n_frames, max(1, n_frames // 10))
    sample   = DS[ECHO_VAR].isel(time=samp_idx).values
    sample   = sample[:, ::PULSE_STEP, :][:, VALID, :]
    log_samp = np.log1p(sample.astype("float32"))
    VMIN, VMAX = float(np.nanmin(log_samp)), float(np.nanmax(log_samp))

    # 256‑entry plasma colormap with alpha ramp
    seg = cm.get_cmap("plasma")(np.linspace(0, 1, 256))
    seg[:, 3] = np.linspace(0, 1, 256)
    COL_TAB = (seg * 255).astype(np.uint8)

    # polar‑to‑Cartesian mapping tables
    N_gate = DS.dims["gate"]
    max_r  = N_gate - 1
    diam   = 2 * max_r + 1
    y_idx, x_idx = np.indices((diam, diam))
    cx = cy = max_r
    dx = x_idx - cx
    dy = cy - y_idx
    r_pix = np.sqrt(dx*dx + dy*dy)

    MASK_MAP      = r_pix <= max_r
    angle_f       = np.arctan2(dy, dx) % (2 * np.pi)
    ANGLE_IDX_MAP = (angle_f / (2*np.pi) * N_theta).astype(np.int32)
    GATE_IDX_MAP  = r_pix.astype(np.int32)

    # GPU flag for optional torch‑mps acceleration
    MPS_OK = use_mps and torch.backends.mps.is_available()

    # I/O thread – one per worker – so PNG encoding never blocks CPU core
    IO_Q = queue.Queue()
    threading.Thread(target=_io_loop, args=(IO_Q,), daemon=True).start()


def _io_loop(q: "queue.Queue[tuple[str, bytes]]"):
    while True:
        path, buf = q.get()
        with open(path, "wb") as f:
            f.write(buf)
        q.task_done()


def fast_log1p(arr: np.ndarray) -> np.ndarray:
    if MPS_OK:
        import torch
        return torch.log1p(torch.from_numpy(arr).to("mps")).cpu().numpy()
    return np.log1p(arr)

# ───── per‑frame rendering function ───────────────────────────────

def render(idx: int):
    timers: dict[str, float] = {}

    # 1) Get angles and mask for this frame
    t0 = time.perf_counter()
    angles = DS[ANGLE_VAR].isel(time=idx).values[::PULSE_STEP]
    valid  = ~np.isnan(angles)
    θ      = np.deg2rad(angles[valid])
    N_theta = θ.size

    sweep = DS[ECHO_VAR].isel(time=idx)[::PULSE_STEP, :].values
    sweep = sweep[valid, :]  # Only use valid pulses for this frame
    timers["load"] = time.perf_counter() - t0

    # 2) Compute per-frame polar-to-Cartesian mapping
    t0 = time.perf_counter()
    N_gate = DS.dims["gate"]
    max_r  = N_gate - 1
    diam   = 2 * max_r + 1
    y_idx, x_idx = np.indices((diam, diam))
    cx = cy = max_r
    dx = x_idx - cx
    dy = cy - y_idx
    r_pix = np.sqrt(dx*dx + dy*dy)

    mask_map = r_pix <= max_r
    angle_f  = np.arctan2(dy, dx) % (2 * np.pi)
    angle_idx_map = (angle_f / (2*np.pi) * N_theta).astype(np.int32)
    gate_idx_map  = r_pix.astype(np.int32)
    timers["mapping"] = time.perf_counter() - t0

    # 3) log1p transform (GPU-accelerated if available)
    t0 = time.perf_counter()
    log_data = fast_log1p(sweep.astype("float32"))
    timers["log1p"] = time.perf_counter() - t0

    # 4) normalise → 0‑255 palette indices
    t0 = time.perf_counter()
    normed   = (log_data - VMIN) / (VMAX - VMIN)
    normed   = np.clip(normed, 0.0, 1.0)
    col_idx  = (normed * 255).astype(np.uint8)
    timers["norm"] = time.perf_counter() - t0

    # 5) polar→Cartesian + colour lookup (using per-frame mappings)
    t0 = time.perf_counter()
    H, W = mask_map.shape
    out  = np.empty((H, W, 4), dtype=np.uint8)
    remap_colormap(H, W, mask_map, angle_idx_map, gate_idx_map, col_idx, COL_TAB, out)
    timers["colormap"] = time.perf_counter() - t0

    # 6) PNG encode (lossless, compression‑0)
    t0 = time.perf_counter()
    buf = io.BytesIO()
    Image.fromarray(out, mode="RGBA").save(buf, format="PNG", compress_level=0)
    timers["png_encode"] = time.perf_counter() - t0

    # 7) enqueue write with informative filename
    t0 = time.perf_counter()
    rcode = RANGE_CODE[idx]
    scode = SCALE_CODE[idx]
    tval  = TIME_COORD[idx]
    gain = GAIN[idx]


    tval = TIME_COORD[idx]
    try:
        # Assume tval is seconds since Unix epoch
        t_dt = datetime.datetime.utcfromtimestamp(float(tval))
        tstr = t_dt.strftime("%Y%m%dT%H%M%S")
    except Exception:
        tstr = f"{float(tval):.0f}"


    fname = f"frame_{idx:05d}_t{tstr}_rc{rcode:02d}_uc{scode}_g{gain}.png"
    IO_Q.put((str(Path(OUT_DIR) / fname), buf.getvalue()))
    timers["queue_put"] = time.perf_counter() - t0

    return idx, timers


# ───── main entry‑point ───────────────────────────────────────────

if __name__ == "__main__":
    global_start = time.perf_counter()

    # open once to discover number of frames, close immediately after
    with xr.open_dataset(FILE) as _ds:
        N_FRAMES = _ds.dims["time"]
    print(f"Total time‑steps to render: {N_FRAMES}")

    # --- ADD THIS BLOCK TO DIAGNOSE VALID ANGLES PER FRAME ---
    import numpy as np
    from scipy import stats

    # Collect counts for all frames
    valid_counts = []
    for idx in range(N_FRAMES):
        angles = _ds[ANGLE_VAR].isel(time=idx).values[::PULSE_STEP]
        n_valid = np.sum(~np.isnan(angles))
        valid_counts.append(n_valid)

    valid_counts = np.array(valid_counts)

    # Calculate statistics
    max_count = np.max(valid_counts)
    mean_count = np.mean(valid_counts)
    median_count = np.median(valid_counts)
    mode_count = stats.mode(valid_counts, keepdims=True)[0][0]  # scipy mode returns an array

    print(f"Valid angle stats across frames:")
    print(f"  Max:    {max_count}")
    print(f"  Mean:   {mean_count:.2f}")
    print(f"  Median: {median_count}")
    print(f"  Mode:   {mode_count}")
    # ---------------------------------------------------------


    timings = []
    with ProcessPoolExecutor(
        initializer=init_worker,
        initargs=(USE_MPS,),
        max_workers=MAX_WORKERS,
    ) as pool:
        futures = [pool.submit(render, i) for i in range(N_FRAMES)]
        for fut in as_completed(futures):
            idx, t = fut.result()
            line = "  ".join(f"{k} {v*1000:.1f} ms" for k, v in t.items())
            print(f"[{idx:05d}] {line}")
            timings.append(t)

    wall = time.perf_counter() - global_start
    print(f"\nFinished {N_FRAMES} frames in {wall:.1f} s")
