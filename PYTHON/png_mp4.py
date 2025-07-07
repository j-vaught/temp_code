#!/usr/bin/env python3
"""png_mp4_adaptive.py – *third‑pass* refactor.

   • Guarantees a **fixed output resolution** (width × height is determined
     from the first overlay PNG).  Every satellite crop is resized to that
     resolution, so MoviePy never encounters mismatched frame sizes – the
     usual culprit behind "garbled after N seconds" H.264 streams.
   • Satellite mosaic margin is now computed from the **true maximum radius**
     including any overlay whose range‑code wasn’t in the lookup table (it
     falls back to 48 NM).
   • Cleaned up caches, removed pad logic (no black borders because we always
     resize the crop).
"""

from __future__ import annotations

import math, re, sys
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr
from PIL import Image

try:
    from moviepy.editor import VideoClip
except ImportError:
    from moviepy.video.VideoClip import VideoClip  # type: ignore

import requests

# ───── USER CONFIG ─────────────────────────────────────────────
NC_FILE       = "/Users/jacobvaught/Downloads/Research_radar_DATA/data/data_9/output.nc"
OVERLAYS_DIR  = "/Users/jacobvaught/Downloads/frames_parallel_9"
OUTPUT_VIDEO  = "output_6.mp4"

ZOOM          = 18   # ArcGIS World‑Imagery zoom level
TILE_SIZE     = 256  # px per tile (ArcGIS fixed)
FPS           = 24
PAD_TILES     = 3    # extra safety margin around max radius (tiles)
FLIP_OVERLAY_Y = True  # Set to True to flip overlay PNGs along horizontal axis
A_SHIFT = 168  # azimuth offset in degrees[THIS ROTATES THE SATELLITE IMAGE, not the PNGS FROM THE RADAR] (e.g. 190° for 10° clockwise from north)

# ───────────────────────────────────────────────────────────────

# range‑code → nautical miles
RANGE_TABLE_NM: Dict[int, float] = {
     0:0.125, 1:0.25, 2:0.5, 3:0.75, 4:1, 5:1.5, 6:2, 7:3, 8:4, 9:6,
    10:8, 11:12, 12:16, 13:24, 14:32, 19:36, 21:0.0625,
    # fallback for any unknown code → 48 NM (large)
    -1:48,
}

# unit‑code → multiplier to convert NM → NM (i.e. relative to NM)
UNIT_FACT: Dict[int, float] = {
    0:1.0,          # NM
    1:0.539957,     # km
    2:0.868976,     # sm
    3:0.493,        # kyd
    # anomalous 496 seen earlier – treat as NM
    496:0.98*1/0.539957,
}

_FILENAME_RE = re.compile(r"frame_(\d+)_t[^_]+_rc(\d+)_uc(\d+)\.png$")

# ───────────── helpers ─────────────────────────────────────────

def read_lat_lon_heading(nc: str) -> Tuple[float,float,float]:
    with xr.open_dataset(nc) as ds:
        return (float(ds.attrs["Metadata_latitude"]),
                float(ds.attrs["Metadata_longitude"]),
                float(ds.attrs["Metadata_heading"]))

def list_overlays(dir_: str) -> List[str]:
    return sorted(str(p) for p in Path(dir_).glob("*.png"))

def parse_overlay(fname: str) -> Tuple[int,int,int]:
    m = _FILENAME_RE.search(fname)
    if not m:
        raise ValueError(f"bad overlay filename: {fname}")
    return tuple(map(int, m.groups()))

def nm_from_codes(range_c: int, unit_c: int) -> float:
    r_nm = RANGE_TABLE_NM.get(range_c, RANGE_TABLE_NM[-1])
    return r_nm * UNIT_FACT.get(unit_c, 1.0)

def tiles_needed(lat: float, radius_nm: float, zoom: int) -> int:
    radius_m = radius_nm * 1852.0
    m_per_px = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    px       = int(math.ceil(radius_m / m_per_px)) * 2
    n_tiles  = math.ceil(px / TILE_SIZE) + PAD_TILES
    return n_tiles | 1  # force odd for symmetry

def fetch_mosaic(lat: float, lon: float, zoom: int, tiles: int) -> Image.Image:
    n        = 2 ** zoom
    x_f      = (lon + 180.0) / 360.0 * n
    y_f      = (1.0 - math.log(math.tan(math.radians(lat)) +
                              (1 / math.cos(math.radians(lat)))) / math.pi) / 2.0 * n
    xc, yc   = int(x_f), int(y_f)
    off      = tiles // 2
    canvas   = Image.new("RGB", (tiles*TILE_SIZE, tiles*TILE_SIZE))
    for row in range(tiles):
        for col in range(tiles):
            xt = xc - off + col
            yt = yc - off + row
            url = ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
                   f"MapServer/tile/{zoom}/{yt}/{xt}")
            resp = requests.get(url, timeout=12)
            resp.raise_for_status()
            tile = Image.open(BytesIO(resp.content))
            canvas.paste(tile, (col*TILE_SIZE, row*TILE_SIZE))
    return canvas

def rotate_north(img: Image.Image, heading: float) -> Image.Image:
    return img.rotate(-heading, expand=True)

def crop_to_radius(img: Image.Image, radius_nm: float, lat: float, zoom: int,
                   out_size: Tuple[int,int]) -> Image.Image:
    rad_m   = radius_nm * 1852.0
    m_per_px= 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    rad_px  = rad_m / m_per_px
    cx, cy  = img.width/2, img.height/2
    box     = (int(cx-rad_px), int(cy-rad_px), int(cx+rad_px), int(cy+rad_px))
    crop    = img.crop(box)
    return crop.resize(out_size, Image.BILINEAR)

# ───────────── frame generator ─────────────────────────────────

def make_frame_gen(rot_bg: Image.Image, overlays: List[str], lat: float,
                    out_wh: Tuple[int,int]):
    cache_bg: Dict[Tuple[int,int], Image.Image] = {}

    def frame(t: float):
        idx = int(t * FPS + 1e-6)
        if idx >= len(overlays):
            idx = len(overlays) - 1
        ov_path = overlays[idx]
        _, rc, uc = parse_overlay(ov_path)
        rad_nm   = nm_from_codes(rc, uc)
        key      = (rc, uc)
        if key not in cache_bg:
            cache_bg[key] = crop_to_radius(rot_bg, rad_nm, lat, ZOOM, out_wh).convert("RGBA")
        bg = cache_bg[key]
        ov = Image.open(ov_path).convert("RGBA").resize(out_wh, Image.NEAREST)
        if FLIP_OVERLAY_Y:
            ov = ov.transpose(Image.FLIP_TOP_BOTTOM)

        frame = Image.alpha_composite(bg, ov).convert("RGB")
        return np.asarray(frame)

    return frame

# ───────────── main ───────────────────────────────────────────

def main():
    lat, lon, heading = read_lat_lon_heading(NC_FILE)
    heading = (heading + A_SHIFT) % 360 ########### CHANGE THIS LINE TO ADD AZIMUTH OFFSET ###########
    overlays = list_overlays(OVERLAYS_DIR)
    if not overlays:
        sys.exit("no overlays found")

    # Determine constant output resolution from first overlay
    w0, h0 = Image.open(overlays[0]).size
    out_wh = (w0, h0)

    # Max radius → tile mosaic
    max_r_nm = max(nm_from_codes(*parse_overlay(p)[1:]) for p in overlays)
    tiles    = tiles_needed(lat, max_r_nm, ZOOM)
    print(f"Max radius {max_r_nm:.2f} NM → {tiles}×{tiles} tiles")

    bg_mosaic = rotate_north(fetch_mosaic(lat, lon, ZOOM, tiles), heading)
    duration  = len(overlays) / FPS

    clip = VideoClip(make_frame_gen(bg_mosaic, overlays, lat, out_wh), duration=duration)
    print(f"Encoding {len(overlays)} frames to {OUTPUT_VIDEO} …")
    clip.write_videofile(OUTPUT_VIDEO, fps=FPS, codec="libx264")

if __name__ == "__main__":
    main()
