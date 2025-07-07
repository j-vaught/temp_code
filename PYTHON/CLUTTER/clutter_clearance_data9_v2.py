"""
radar_clean_pipeline.py
======================
Minimal offline processing for marine FMCW radar frames (transparent PNG).

Implemented speckle‑reduction blocks — toggle each with a Boolean flag:
    2.1  Multi‑look averaging (median/mean)
    2.2  Lee adaptive filter
    2.3  BM3D collaborative filtering  (requires `pip install bm3d`)
    2.4  Noise2Void self‑supervised   (requires `pip install n2v tensorflow` + trained model)

Optionally fuse cleaned images from several gain settings with OpenCV’s
MergeMertens exposure‑fusion to obtain a high‑dynamic‑range composite.

Usage
-----
    python radar_clean_pipeline.py

Basic deps: numpy, opencv‑python‑headless; optional: bm3d, n2v, tensorflow.
"""
from __future__ import annotations
import os, re, glob
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import cv2                    # type: ignore
import numpy as np            # type: ignore

# ---------- optional imports (silently handled) ----------
try:
    import bm3d               # type: ignore  # 2.3
except ImportError:
    bm3d = None
try:
    from n2v.models import N2V # type: ignore  # 2.4
except ImportError:
    N2V = None

# =========================== USER CONFIGURATION ============================
DATA_DIR   = "/Users/jacobvaught/Downloads/frames_parallel_9g"   # <‑‑ set your dataset rootis there
OUTPUT_DIR = "Z_clutter_output"

# Toggle individual blocks --------------------------------------------------
DO_MULTI_LOOK = False   # 2.1  (median by default)
DO_LEE_FILTER = False  # 2.2
DO_BM3D       = True  # 2.3
DO_HDR_FUSION = False   # fuse cleaned frames across gains

# Internal params -----------------------------------------------------------
MAX_FRAMES_PER_GAIN = 4   # the four looks per gain
LEE_WIN_SIZE        = 21   # odd window size
LEE_DAMPING         = 3.0 # 1 ⇒ classic Lee; >1 smoother
MERTENS_WEIGHTS     = (1.0, 1.0, 0.2)  # (contrast, saturation, exposure)
USER_GAINS          = [80, 90, 99]   # gains to include in HDR fusion
USER_RANGES         = [3]  # e.g. [3] to restrict range‑codes; None ⇒ all

# =============================== UTILITIES =================================

def safe_write(out_path: Path, img: np.ndarray) -> None:
    """Save PNG, appending a numeric suffix if the file exists."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem, ext = out_path.stem, out_path.suffix
    counter   = 1
    while out_path.exists():
        out_path = out_path.with_name(f"{stem}_{counter}{ext}")
        counter += 1
    cv2.imwrite(str(out_path), img)

# Accept timestamps like 19700101T000001
FNAME_RE = re.compile(
    r"frame_(?P<idx>\d+)_t(?P<ts>[0-9T]+)_rc(?P<rc>\d+)_uc(?P<uc>\d+)_g(?P<gain>[\d\.]+)\.png$",
    re.IGNORECASE,
)

def parse_fname(fname: Path) -> Dict[str, Any]:
    m = FNAME_RE.search(fname.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {fname}")
    return {
        "idx":  int(m.group("idx")),
        "ts":   m.group("ts"),
        "rc":   int(m.group("rc")),
        "uc":   int(m.group("uc")),
        "gain": float(m.group("gain")),
    }

# ====================== SPECKLE‑REDUCTION BLOCKS ==========================

# 2.1 ----------------------------------------------------------------------
def multi_look_average(imgs: List[np.ndarray], reducer: str = "median") -> np.ndarray:
    stack = np.stack(imgs, axis=0).astype(np.float32)
    out   = np.median(stack, axis=0) if reducer == "median" else np.mean(stack, axis=0)
    return np.clip(out, 0, 255).astype(np.uint8)

# 2.2 ----------------------------------------------------------------------
def lee_filter(img: np.ndarray, win: int = LEE_WIN_SIZE, damping: float = LEE_DAMPING) -> np.ndarray:
    if win % 2 == 0:
        win += 1
    img_f   = img.astype(np.float32)
    local_m = cv2.boxFilter(img_f, -1, (win, win))
    local_v = cv2.boxFilter(img_f ** 2, -1, (win, win)) - local_m ** 2
    noise_v = np.median(local_v)
    gain    = (local_v / (local_v + noise_v + 1e-8)) ** damping
    out     = local_m + gain * (img_f - local_m)
    return np.clip(out, 0, 255).astype(np.uint8)

# 2.3 ----------------------------------------------------------------------
def bm3d_denoise(img: np.ndarray, sigma_psd: float = 200/255) -> np.ndarray:
    if bm3d is None:
        raise ImportError("bm3d not installed.  pip install bm3d")
    out_f = bm3d.bm3d(img.astype(np.float32) / 255.0, sigma_psd=sigma_psd)
    return np.clip(out_f * 255, 0, 255).astype(np.uint8)


# ============================ HDR FUSION ==================================

def hdr_fusion(imgs_gray: List[np.ndarray]) -> np.ndarray:
    merge = cv2.createMergeMertens(*MERTENS_WEIGHTS)
    imgs_f = [im.astype(np.float32) / 255.0 for im in imgs_gray]
    fused  = merge.process(imgs_f)
    return np.clip(fused * 255, 0, 255).astype(np.uint8)

# ============================ FILE I/O HELPERS ============================

def load_frames(root: str) -> List[Tuple[Path, Dict[str, float]]]:
    paths = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
    return [(Path(p), parse_fname(Path(p))) for p in paths]


def group_by_gain_range(frames: List[Tuple[Path, Dict[str, float]]]):
    grp: Dict[int, Dict[float, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for path, meta in frames:
        grp[meta["rc"]][meta["gain"]].append(path)
    return grp

# ============================= CORE PROCESS ===============================

def process_gain_group(rc: int, gain: float, paths: List[Path]):
    imgs_rgba = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths[:MAX_FRAMES_PER_GAIN]]
    imgs_gray = [cv2.cvtColor(im, cv2.COLOR_BGRA2GRAY) for im in imgs_rgba]
    a_channel = imgs_rgba[0][:, :, 3]

    # ----- 2.1 -----
    result = multi_look_average(imgs_gray) if DO_MULTI_LOOK else imgs_gray[0]

    # ----- 2.2 -----
    if DO_LEE_FILTER:
        print(f"Applying Lee filter to: {paths[0]}")
        result = lee_filter(result)

    # ----- 2.3 -----
    if DO_BM3D:
        result = bm3d_denoise(result)

    safe_write(Path(OUTPUT_DIR) / f"rc{rc}_g{int(gain)}.png",
               cv2.merge([result, result, result, a_channel]))

    return result

# ================================ MAIN ====================================

def main():
    frames = load_frames(DATA_DIR)
    if not frames:
        raise SystemExit("No PNG frames found. Check DATA_DIR.")

    grp_by_rc_gain = group_by_gain_range(frames)

    for rc, gains in grp_by_rc_gain.items():
        if USER_RANGES and rc not in USER_RANGES:
            continue

        cleaned_first_frames: List[np.ndarray] = []

        for gain in USER_GAINS if USER_GAINS else gains.keys():
            paths = gains.get(gain, [])
            if not paths:
                continue
            result = process_gain_group(rc, gain, paths)
            if DO_HDR_FUSION:
                cleaned_first_frames.append(result)

        if DO_HDR_FUSION and len(cleaned_first_frames) >= 2:
            fused = hdr_fusion(cleaned_first_frames)
            safe_write(Path(OUTPUT_DIR) / f"rc{rc}_hdr.png",
                       cv2.merge([fused, fused, fused, np.full_like(fused, 255)]))

    print("Done ✔  Output in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
