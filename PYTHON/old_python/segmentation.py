import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys

# ==== CONFIG ====
ONNX_PATH  = Path("/Users/jacobvaught/Downloads/WaterSegNets/onnx_files/"
                  "model_Unet_mobilenet_v2_DiceLoss_best_model40.onnx")
IMAGE      = Path("/Users/jacobvaught/Downloads/test_image.png")
RESIZE_TO  = (256, 256)
THRESHOLD  = 0.5
TILE_ROWS  = 4
TILE_COLS  = 4

# ==== DEBUG HELPERS ====
def debug(var, name):
    if isinstance(var, np.ndarray):
        print(f"[DEBUG] {name:20s} | shape: {var.shape} | dtype: {var.dtype}")
    else:
        print(f"[DEBUG] {name:20s} | value: {var}")

def fatal(msg):
    print(f"[FATAL] {msg}", file=sys.stderr)
    sys.exit(1)

# ==== 1) VERIFY FILES ====
if not ONNX_PATH.exists(): fatal(f"ONNX not found: {ONNX_PATH}")
if not IMAGE.exists():     fatal(f"Image not found: {IMAGE}")

# ==== 2) INIT ONNX SESSION ====
session  = ort.InferenceSession(str(ONNX_PATH))
inp_name = session.get_inputs()[0].name

# ==== 3) LOAD IMAGE & SETUP ====
orig = cv2.imread(str(IMAGE))
if orig is None: fatal("cv2.imread failed")
h, w = orig.shape[:2]
half_h = h // 2     # midpoint in pixels
debug(orig, "orig")

# precompute tile sizes (assume h,w divisible by 4 for simplicity)
tile_h = h // TILE_ROWS
tile_w = w // TILE_COLS
half_rows = TILE_ROWS // 2   # number of tile-rows in the top half

# ==== 4) WHOLE-IMAGE / “1×1” MASK ====
# preprocess
rgb_whole   = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
resized_wh  = cv2.resize(rgb_whole, RESIZE_TO)
tensor_wh   = (resized_wh.astype(np.float32)/255.0).transpose(2,0,1)[None,...]
# inference
out_wh      = session.run(None, {inp_name: tensor_wh})[0][0,0]
mask_wh256  = (out_wh > THRESHOLD).astype(np.uint8) * 255
# upsample to full size
mask_whole_fullres = cv2.resize(mask_wh256, (w,h), interpolation=cv2.INTER_NEAREST)
debug(mask_whole_fullres, "mask_whole_fullres")

# ==== 5) TILED / “4×4” MASK FOR TOP HALF ====
mask_tiled = np.zeros((h, w), dtype=np.uint8)

for i in range(half_rows):        # only rows 0 and 1
    for j in range(TILE_COLS):
        y0, y1 = i*tile_h, (i+1)*tile_h
        x0, x1 = j*tile_w, (j+1)*tile_w
        tile = orig[y0:y1, x0:x1]

        # preprocess tile
        rgb_t = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        rsz_t = cv2.resize(rgb_t, RESIZE_TO)
        t_ten = (rsz_t.astype(np.float32)/255.0).transpose(2,0,1)[None,...]

        # inference
        out_t  = session.run(None, {inp_name: t_ten})[0][0,0]
        m256   = (out_t > THRESHOLD).astype(np.uint8) * 255
        debug(m256, f"mask_tile[{i},{j}]")

        # upsample back to tile-size
        m_full = cv2.resize(m256, (tile_w, tile_h),
                            interpolation=cv2.INTER_NEAREST)

        # place into top half of mask_tiled
        mask_tiled[y0:y1, x0:x1] = m_full

debug(mask_tiled, "mask_tiled (top half)")

# ==== 6) COMBINE TOP & BOTTOM ====
final_mask = np.zeros((h, w), dtype=np.uint8)
# top half from tiled
final_mask[:half_h, :] = mask_tiled[:half_h, :]
# bottom half from whole-image
final_mask[half_h:, :] = mask_whole_fullres[half_h:, :]
debug(final_mask, "final_mask")

# save the mixed mask
cv2.imwrite("mask_mixed.png", final_mask)
print("[OK] mask_mixed.png written")

# ==== 7) OVERLAY & SAVE ====
overlay = orig.copy()
overlay[final_mask == 255] = (0, 255, 255)   # cyan
cv2.imwrite("overlay_mixed.png", overlay)
print("[OK] overlay_mixed.png written")

print("\nDone!  Check mask_mixed.png & overlay_mixed.png.")
