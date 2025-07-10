"""
radar_clutter_processing.py

Offline processing pipeline for marine FMCW radar frames stored as transparent PNGs.
Each filename must follow the pattern:
    frame_<idx>_t<unix_time>_rc<range_code>_uc<unit_code>_g<gain>.png

The script offers simple clutter‑removal options and an optional multi‑gain HDR fusion.
Enable/disable techniques via the bool flags below, set DATA_DIR to your dataset root,
then run:
    python radar_clutter_processing.py

Outputs are written to ./test_clutter_output.
If a filename already exists, an incrementing suffix is appended.

Dependencies (install via pip):
    numpy, opencv‑python‑headless (or opencv‑python for GUI), pywavelets, tqdm, scikit‑image
Optional for RPCA:  scikit‑learn, numpy‑linalg‑lstsq (or the 'r_pca' package if installed)
pip install numpy opencv-python-headless pywavelets tqdm scikit-image scikit-learn numpy

"""

from __future__ import annotations
import os, re, glob, uuid, shutil, itertools, math
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2                    # type: ignore
import numpy as np            # type: ignore
import pywt                   # type: ignore
from skimage import exposure  # type: ignore
from tqdm import tqdm         # type: ignore

# =========================== USER CONFIGURATION ============================
DATA_DIR      = "/Users/jacobvaught/Downloads/test"     # <‑‑‑‑‑‑‑‑ set before running
OUTPUT_DIR    = "test_clutter_output_HDR"

# Toggle individual processing steps
DO_MTI_SUBTRACT  = False   # temporal median (MTI‑like) clutter suppression
DO_WAVELET       = False   # wavelet denoising
DO_RPCA          = False  # low‑rank (RPCA) background subtraction
DO_HDR_FUSION    = False    # multi‑gain exposure fusion (Mertens)
DO_MULTI_FRAME_DENOISE = True # multi‑frame denoising (OpenCV NLM)
DO_mGAIN_DENOISE = False  # multi‑gain denoising (across all gains for a range code)


# ---------------------------------------------------------------------------
# Internal constants – adjust only if you need to tweak behaviour
MAX_FRAMES_PER_GAIN = 2          # expect 2 frames for each gain / range combo
N_SMART_GAINS       = 1          # how many gains to pick automatically
WAVELET             = "db2"      # mother wavelet for denoising
WAVELET_LEVEL       = 6          # decomposition level
MERGE_MERTENS_CONTRAST_WEIGHT = 0.1  # OpenCV Mertens parameters
MERGE_MERTENS_SATURATION_WEIGHT = 0.1
MERGE_MERTENS_EXPOSURE_WEIGHT = 3
USER_GAINS = [30]  # <--- set your desired gains here, or None for all
USER_RANGES = [1]



# =============================== UTILITIES =================================

def safe_write(out_path: Path, img: np.ndarray) -> None:
    """Save PNG, appending numeric suffix if file exists."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem, ext = out_path.stem, out_path.suffix
    counter   = 1
    while out_path.exists():
        out_path = out_path.with_name(f"{stem}_{counter}{ext}")
        counter += 1
    cv2.imwrite(str(out_path), img)

# Accept timestamps consisting of digits and the letter T (e.g. 19700101T000001)
FNAME_RE = re.compile(
    r"frame_(?P<idx>\d+)_t(?P<ts>[0-9T]+)_rc(?P<rc>\d+)_uc(?P<uc>\d+)_g(?P<gain>[\d\.]+)\.png$",
    re.IGNORECASE,
)

def parse_fname(fname: Path) -> Dict[str, Any]:
    m = FNAME_RE.search(fname.name)
    if not m:
        raise ValueError(f"Filename does not match expected pattern: {fname}")
    return {
        "idx":   int(m.group("idx")),
        "ts":    m.group("ts"),         # <--- do NOT call int() here!
        "rc":    int(m.group("rc")),
        "uc":    int(m.group("uc")),
        "gain":  float(m.group("gain")),
    }


# ============================ LOADING FRAMES ===============================

def load_frames(root: str) -> List[Tuple[Path, Dict[str,float]]]:
    paths = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
    frames = []
    for p in paths:
        meta = parse_fname(Path(p))
        frames.append((Path(p), meta))
    if not frames:
        raise RuntimeError("No PNG frames found – check DATA_DIR path.")
    return frames


def group_by_gain_range(frames: List[Tuple[Path, Dict[str,float]]]):
    """Return nested dict {(rc): {gain: [Path, ...]}}"""
    grp: Dict[int, Dict[float, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for path, meta in frames:
        grp[meta["rc"]][meta["gain"]].append(path)
    return grp

# ======================= CLUTTER‑REMOVAL METHODS ===========================

def mti_subtract(imgs: List[np.ndarray]) -> List[np.ndarray]:
    """Subtract median background (simple n‑pulse MTI)."""
    stack = np.stack(imgs, axis=0)
    bg    = np.median(stack, axis=0)
    out   = [np.clip((f.astype(np.float32) - bg), 0, 255).astype(np.uint8) for f in stack]
    return out


def wavelet_denoise(img: np.ndarray) -> np.ndarray:
    coeffs = pywt.wavedec2(img, WAVELET, level=WAVELET_LEVEL)
    sigma  = np.median(np.abs(coeffs[-1][0])) / 0.6745  # noise estimate
    uthresh = sigma * math.sqrt(2 * math.log(img.size))
    coeffs_thresh = list(coeffs)
    coeffs_thresh[1:] = [tuple(pywt.threshold(c, value=uthresh, mode="soft") for c in level)
                         for level in coeffs[1:]]
    recon = pywt.waverec2(coeffs_thresh, WAVELET)
    # ---- Robust shape, nan, and type fixing ----
    recon = np.nan_to_num(recon, nan=0.0, posinf=255, neginf=0)
    recon = np.clip(recon, 0, 255)
    if recon.shape != img.shape:
        # This can rarely happen due to padding in wavelet
        recon = cv2.resize(recon, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    return recon.astype(np.uint8)



def rpca_subtract(imgs: List[np.ndarray]) -> List[np.ndarray]:
    try:
        import sys
        sys.path.append("/Users/jacobvaught/Downloads/robust-pca")
        from r_pca import R_pca
  # pip install r_pca
    except ImportError:
        raise ImportError("RPCA step requested but 'r_pca' library not installed.")
    stack = np.stack([i.reshape(-1) for i in imgs], axis=1).astype(np.float32)
    rpca = R_pca(stack)
    L, S = rpca.fit(max_iter=1000, iter_print=False)
    sparse_imgs = [np.clip(np.abs(S[:,k]).reshape(imgs[0].shape), 0, 255).astype(np.uint8)
                   for k in range(S.shape[1])]
    return sparse_imgs

# ============================ HDR FUSION ===================================

def hdr_fusion(gain_imgs: List[np.ndarray]) -> np.ndarray:
    """Fuse multiple gain images using OpenCV Mertens exposure fusion."""
    merge = cv2.createMergeMertens(MERGE_MERTENS_CONTRAST_WEIGHT,
                                   MERGE_MERTENS_SATURATION_WEIGHT,
                                   MERGE_MERTENS_EXPOSURE_WEIGHT)
    # OpenCV expects float32 [0..1]
    imgs_f = [img.astype(np.float32) / 255.0 for img in gain_imgs]
    fused  = merge.process(imgs_f)
    fused8 = np.clip(fused*255, 0, 255).astype(np.uint8)
    return fused8

# ============================ Muti-frame denoise ===================================

def multi_frame_denoise(
    imgs_gray: List[np.ndarray], 
    h: int = 30,
    templateWindowSize: int = 9,
    searchWindowSize: int = 31,
    use_frames: int = None
) -> np.ndarray:
    n = len(imgs_gray)
    if n < 3:
        print(f"Skipping multi-frame denoise: need at least 3 frames, got {n}.")
        return imgs_gray[0]

    # Decide number of frames to use (must be odd, at least 3)
    if use_frames is not None and use_frames < n:
        n = use_frames
    if n % 2 == 0:
        n -= 1
    if n < 3:
        n = 3

    # Use only the first n frames
    imgs_used = imgs_gray[:n]
    srcs = np.stack(imgs_used, axis=0).astype(np.uint8)
    height, width = srcs[0].shape
    templateWindowSize = min(templateWindowSize, height, width)
    searchWindowSize = min(searchWindowSize, height, width)
    if templateWindowSize % 2 == 0:
        templateWindowSize -= 1
    if searchWindowSize % 2 == 0:
        searchWindowSize -= 1

    # Denoise the middle frame, as required by OpenCV API
    mid = n // 2

    print(f"srcs.shape={srcs.shape}, srcs.dtype={srcs.dtype}")
    print(f"templateWindowSize={templateWindowSize}, searchWindowSize={searchWindowSize}, num frames={n}, image shape={srcs[0].shape}, denoise index={mid}")

    try:
        denoised = cv2.fastNlMeansDenoisingMulti(
            srcs,
            imgToDenoiseIndex=mid,
            temporalWindowSize=n,
            dst=None,
            h=h,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize
        )
    except cv2.error as e:
        print("OpenCV error:", e)
        print("Falling back to first frame")
        return imgs_gray[0]
    return denoised

def multi_gain_denoise(
    grp_by_rc_gain: dict,
    rc: int,
    user_gains: list,
    max_frames_per_gain: int = 1,
    h: int = 30,
    templateWindowSize: int = 9,
    searchWindowSize: int = 31
):
    """
    For a given range code (rc) and list of gains,
    combine up to max_frames_per_gain frames per gain (across all gains) into one list,
    and run multi-frame denoising ONCE on the whole stack.
    """
    all_paths = []
    all_imgs_rgba = []

    # --- DEBUG: print header ---
    print(f"\n[DEBUG] multi_gain_denoise: rc={rc}, max_frames_per_gain={max_frames_per_gain}")
    # --------------------------------

    for gain in user_gains:
        gain_frames = grp_by_rc_gain.get(rc, {}).get(gain, [])
        # Use up to max_frames_per_gain from this gain
        selected = gain_frames[:max_frames_per_gain]

        # --- DEBUG: print per-gain selection ---
        print(f"[DEBUG]   gain={gain:>5} → selecting {len(selected)} frame(s):")
        for p in selected:
            print(f"           {p.name}")
        # ----------------------------------------

        all_paths.extend(selected)
        all_imgs_rgba.extend([cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in selected])

    if len(all_imgs_rgba) < 3:
        print(f"[DEBUG]   Not enough frames for rc={rc} (got {len(all_imgs_rgba)}). Skipping.")
        return

    # Now you’ll see exactly which three (or however many) PNGs
    # are about to be converted to grayscale and smashed into the stack.
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in all_imgs_rgba]

    # Run multi-frame denoise on all combined frames (regardless of gain)
    denoised = multi_frame_denoise(
        imgs_gray,
        h=h,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize
    )
    out_name = f"rc{rc}_allgains_multi_denoise.png"
    out_rgba = cv2.merge([denoised, denoised, denoised, all_imgs_rgba[0][:, :, 3]])
    safe_write(Path(OUTPUT_DIR) / out_name, out_rgba)
    print(f"Saved {out_name}")

# ============================= MAIN DRIVER =================================

def smart_gain_selection(grp: Dict[float, List[Path]]) -> List[float]:
    """Return either user-selected gains (if USER_GAINS is not None), or N spread across available."""
    gains_sorted = sorted(grp.keys())
    if USER_GAINS is not None:
        # Only keep gains present in both the dataset and the user list.
        return [g for g in gains_sorted if g in USER_GAINS]
    if len(gains_sorted) <= N_SMART_GAINS:
        return gains_sorted
    idxs = np.linspace(0, len(gains_sorted)-1, N_SMART_GAINS, dtype=int)
    return [gains_sorted[i] for i in idxs]



def process_gain_group(rc_gain: Tuple[int,float], paths: List[Path]):
    rc, gain = rc_gain
    # Load frames (expect <= MAX_FRAMES_PER_GAIN)
    imgs_rgba = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in paths[:MAX_FRAMES_PER_GAIN]]
    # Split alpha and intensity (assume grayscale in RGB + alpha)
    imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in imgs_rgba]

    # --------------- MTI median subtraction -----------------
    if DO_MTI_SUBTRACT:
        out_imgs = mti_subtract(imgs_gray)
        for path, out in zip(paths, out_imgs):
            out_rgba = cv2.merge([out, out, out, imgs_rgba[0][:,:,3]])
            safe_write(Path(OUTPUT_DIR) / f"{path.stem}_mti.png", out_rgba)
    # --------------- Wavelet denoise -----------------------
    if DO_WAVELET:
        for path, img in zip(paths, imgs_gray):
            out = wavelet_denoise(img)
            out_rgba = cv2.merge([out, out, out, imgs_rgba[0][:,:,3]])
            if (out.shape == imgs_rgba[0][:,:,3].shape and
                out.dtype == np.uint8 and
                np.isfinite(out).all()):
                safe_write(Path(OUTPUT_DIR) / f"{path.stem}_wav.png",
                        cv2.merge([out, out, out, imgs_rgba[0][:, :, 3]]))
            else:
                print(f"Skipped {path} due to output shape/type/nan issues after denoising.")

    # --------------- RPCA ----------------------------------
    if DO_RPCA:
        try:
            out_imgs = rpca_subtract(imgs_gray)
            for path, out in zip(paths, out_imgs):
                out_rgba = cv2.merge([out, out, out, imgs_rgba[0][:,:,3]])
                safe_write(Path(OUTPUT_DIR) / f"{path.stem}_rpca.png", out_rgba)
        except ImportError as e:
            print(e)
    # ------------- Multi-frame denoising ---------------
    if DO_MULTI_FRAME_DENOISE:
        denoised = multi_frame_denoise(imgs_gray)
        out_name = f"rc{rc}_g{int(gain)}_multi_denoise.png"
        out_rgba = cv2.merge([denoised, denoised, denoised, imgs_rgba[0][:,:,3]])
        safe_write(Path(OUTPUT_DIR) / out_name, out_rgba)


# ============================= ENTRY‑POINT ================================

def main():
    frames = load_frames(DATA_DIR)
    grp_by_rc_gain = group_by_gain_range(frames)

    # Select smart gains across all range codes
    for rc, gains in grp_by_rc_gain.items():
        if USER_RANGES is not None and rc not in USER_RANGES:
            continue  # skip this range code
        
        if DO_mGAIN_DENOISE:
            multi_gain_denoise(
                grp_by_rc_gain,
                rc,
                USER_GAINS,
                max_frames_per_gain=1,  # you can use 2, 3, etc.
                h=30,
                templateWindowSize=9,
                searchWindowSize=31
            )
        
        selected_gains = smart_gain_selection(gains)
        for gain in selected_gains:
            paths = gains[gain]
            process_gain_group((rc, gain), paths)

        # HDR fusion across gains (first frame of each gain)
        if DO_HDR_FUSION and len(selected_gains) >= 2:
            base_paths = [gains[g][0] for g in selected_gains]
            print(f"\n[HDR FUSION] Range code rc={rc}")
            print("  Using the following gains and frames:")
            for g, p in zip(selected_gains, base_paths):
                print(f"    gain={g}: {p.name}")
            print("  Starting HDR fusion...")

            imgs_rgba = [cv2.imread(str(p), cv2.IMREAD_UNCHANGED) for p in base_paths]
            imgs_gray = [cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) for img in imgs_rgba]
            fused_gray = hdr_fusion(imgs_gray)
            fused_rgba = cv2.merge([fused_gray, fused_gray, fused_gray, imgs_rgba[0][:,:,3]])
            fused_rgba[:,:,:3] = 255 - fused_rgba[:,:,:3]
            out_name = f"rc{rc}_hdr.png"
            safe_write(Path(OUTPUT_DIR) / out_name, fused_rgba)
            print(f"  Saved HDR-fused output as: {out_name}")
            print(f"    Output shape: {fused_rgba.shape}, dtype: {fused_rgba.dtype}")
            print("-" * 40)




    print("Processing complete. Output in", OUTPUT_DIR)

if __name__ == "__main__":
    main()
