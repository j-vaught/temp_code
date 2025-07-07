

#!/usr/bin/env python3
"""multi_polygon_ramp.py

Draw multiple polygons and export a **ramped colour‑mask**:
  * **Green** (0 ,255 ,0) outside your marks.
  * **Red**   (0 ,0 ,255) inside your marks.
  * Smooth *ramp* from green→red over *ramp_px* pixels, with rounded edges.
  * Constant 0.5 global alpha (BGRA PNG).

While drawing, you see only the red feathered preview and polygon outlines.

Controls
--------
LEFT CLICK   – add a vertex.
RIGHT CLICK  – finalise current polygon (or press **f** / ENTER).
**f** / ENTER – finalise current polygon.
**r**        – reset all.
**s**        – save mask and exit.
ESC          – exit without saving.
"""

from __future__ import annotations
import cv2
import numpy as np
import argparse
from pathlib import Path

# ------------------------------------------------------------
# Core interactive function
# ------------------------------------------------------------
def interactive_mask_overlay(
    img: np.ndarray,
    out_path: Path | str,
    ramp_px: int = 10,
    alpha: float = 0.5,
):
    # preview setup
    max_w, max_h = 800, 600
    H, W = img.shape[:2]
    scale0 = min(max_w / W, max_h / H, 1.0)
    preview = cv2.resize(img, None, fx=scale0, fy=scale0, interpolation=cv2.INTER_AREA)

    # polygon data
    polys_full: list[list[tuple[int,int]]] = []
    polys_prev: list[list[tuple[int,int]]] = []
    current_f: list[tuple[int,int]] = []
    current_p: list[tuple[int,int]] = []

    window = "Ramp Mask"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # preview builder
    def build_preview_base() -> np.ndarray:
        if not polys_prev:
            return preview.copy()
        mask_prev = np.zeros(preview.shape[:2], np.uint8)
        for pts in polys_prev:
            cv2.fillPoly(mask_prev, [np.array(pts, np.int32)], 255)
        sigma = max(ramp_px * scale0 / 3, 1.0)
        feather = cv2.GaussianBlur(mask_prev, (0,0), sigma).astype(float)/255.0
        overlay = preview.copy().astype(float)
        red = np.zeros_like(preview); red[:,:,2] = 255
        overlay = overlay*(1-feather[...,None]) + red*feather[...,None]
        return overlay.astype(np.uint8)

    # redraw GUI
    def redraw():
        disp = build_preview_base()
        for pts in polys_prev:
            cv2.polylines(disp, [np.array(pts, np.int32)], True, (0,255,0), 1)
        for i,p in enumerate(current_p):
            cv2.circle(disp, p, 3, (0,255,0), -1)
            if i>0: cv2.line(disp, current_p[i-1], p, (0,255,0),1)
        cv2.imshow(window, disp)

    # finalise current polygon
    def finalise():
        if len(current_f) >= 3:
            polys_full.append(current_f.copy())
            polys_prev.append(current_p.copy())
        else:
            print("Need ≥3 points to finalise a polygon.")
        current_f.clear(); current_p.clear(); redraw()

    # save mask function (hoisted before loop)
    def save_mask():
        if not polys_full:
            print("No polygons to save.")
            return
        mask = np.zeros((H, W), np.uint8)
        for pts in polys_full:
            cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
        dist_out = cv2.distanceTransform(255-mask, cv2.DIST_L2,5)
        dist_in  = cv2.distanceTransform(mask, cv2.DIST_L2,5)
        signed   = dist_out - dist_in
        f        = np.clip((-signed + ramp_px)/(2*ramp_px),0,1)[...,None]
        green = np.array([0,255,0],np.float32)
        red   = np.array([0,0,255],np.float32)
        colour = (1-f)*green + f*red
        alpha_ch = np.full((H,W,1), int(alpha*255),np.uint8)
        mask_bgra = np.concatenate([colour.astype(np.uint8), alpha_ch], axis=2)
        cv2.imwrite(str(out_path), mask_bgra)
        print(f"✅ Saved ramp mask to {out_path}")

    # mouse callback
    def on_mouse(event,x,y,flags,_):
        if event==cv2.EVENT_LBUTTONDOWN:
            xf,yf = int(x/scale0), int(y/scale0)
            current_f.append((xf,yf)); current_p.append((x,y)); redraw()
        elif event==cv2.EVENT_RBUTTONDOWN:
            finalise()

    cv2.setMouseCallback(window,on_mouse)
    redraw()

    # main loop
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('f'),13): finalise()
        elif k==ord('r'): polys_full.clear(); polys_prev.clear(); current_f.clear(); current_p.clear(); redraw()
        elif k==ord('s'): save_mask(); break
        elif k==27: print("Cancelled – no save."); break

    cv2.destroyAllWindows()

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p=argparse.ArgumentParser(description="Interactive ramp mask drawer.")
    p.add_argument("--input_image",type=Path,default=Path("overlay_mixed.png"),help="Preview background.")
    p.add_argument("--output",type=Path,default=Path("mask_ramp.png"),help="BGRA ramp mask.")
    p.add_argument("--ramp",type=int,default=80,help="Ramp width px.")
    p.add_argument("--alpha",type=float,default=0.5,help="Global alpha.")
    return p.parse_args()

if __name__=="__main__":
    args=parse_args()
    img=cv2.imread(str(args.input_image))
    if img is None: raise SystemExit(f"❌ Could not load {args.input_image}")
    interactive_mask_overlay(img, args.output, ramp_px=args.ramp, alpha=args.alpha)
