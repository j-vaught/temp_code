#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Random-pulse radar simulator showing only the top-right quadrant (0°–90°),
with custom facecolor processing: user-specified red cell, inner ring
and outer ring transparencies, plus default transparency for all others.
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import os

# ── User parameters ─────────────────────────────────────────────────────────
R_MAX               = 21.0      # metres
N_PULSES            = 36        # pulses per full 360° rotation
N_CELLS             = 11        # range cells per pulse
BASE_CMAP           = 'viridis' # base colormap for values > 0
c_min, c_max        = 0.0, 1.0

# ── Specify red cell (indices within the quarter slice) ────────────────────
RED_PULSE           = 4         # azimuth index [0..N_QUARTER-1]
RED_CELL            = 5         # range-cell index [0..N_CELLS-1]

# ── Specify inner ring window sizes and transparency ───────────────────────
N_ALPHA_PULSES      = 1        # half-width in pulse direction
N_ALPHA_CELLS       = 1         # half-width in cell (range) direction
ALPHA_INNER         = 0.05       # transparency for inner-ring neighbors

# ── Specify outer ring window sizes and transparency ───────────────────────
N_OUTER_PULSES      = 3         # half-width in pulse direction for outer ring
N_OUTER_CELLS       = 3         # half-width in cell direction for outer ring
ALPHA_OUTER         = 1.0       # transparency for outer-ring neighbors

# ── Default transparency for all other cells ──────────────────────────────
ALPHA_DEFAULT       = 0.4       # e.g. fully opaque

# ── Generate data for quarter pulses ───────────────────────────────────────
N_QUARTER = N_PULSES // 4
P_full    = np.random.rand(N_PULSES, N_CELLS)
P         = P_full[:N_QUARTER, :]

# ── Build radial & angular grids for quarter ──────────────────────────────
theta_edges = np.linspace(0, np.pi/2, N_QUARTER + 1)
rng_edges   = np.linspace(0, R_MAX,       N_CELLS + 1)

# ── Map P through base colormap to RGBA facecolors ────────────────────────
base_cmap  = plt.cm.get_cmap(BASE_CMAP)
norm       = colors.Normalize(vmin=c_min, vmax=c_max)
# RGBA array shape (N_QUARTER, N_CELLS, 4)
rgba       = base_cmap(norm(P))
# Transpose to shape (N_CELLS, N_QUARTER, 4) matching pcolormesh grid
facecolors = rgba.transpose((1, 0, 2))
# Initialize all alpha to default
facecolors[:, :, 3] = ALPHA_DEFAULT

# ── Apply inner-ring transparency ─────────────────────────────────────────
for dp in range(-N_ALPHA_PULSES, N_ALPHA_PULSES + 1):
    for dc in range(-N_ALPHA_CELLS, N_ALPHA_CELLS + 1):
        ip = RED_PULSE + dp
        ic = RED_CELL + dc
        if dp == 0 and dc == 0:
            continue  # center is red cell, handled later
        if 0 <= ip < N_QUARTER and 0 <= ic < N_CELLS:
            facecolors[ic, ip, 3] = ALPHA_INNER

# ── Apply outer-ring transparency ─────────────────────────────────────────
for dp in range(-N_OUTER_PULSES, N_OUTER_PULSES + 1):
    for dc in range(-N_OUTER_CELLS, N_OUTER_CELLS + 1):
        ip = RED_PULSE + dp
        ic = RED_CELL + dc
        # skip center and inner ring
        if dp == 0 and dc == 0:
            continue
        if abs(dp) <= N_ALPHA_PULSES and abs(dc) <= N_ALPHA_CELLS:
            continue
        if 0 <= ip < N_QUARTER and 0 <= ic < N_CELLS:
            facecolors[ic, ip, 3] = ALPHA_OUTER

# ── Color the red cell ─────────────────────────────────────────────────────
if 0 <= RED_PULSE < N_QUARTER and 0 <= RED_CELL < N_CELLS:
    facecolors[RED_CELL, RED_PULSE] = [1.0, 0.0, 0.0, 1.0]

# ── Plot polar PPI ────────────────────────────────────────────────────────
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8, 8))
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_thetamin(0)
ax.set_thetamax(90)

ax.pcolormesh(
    theta_edges,
    rng_edges,
    facecolors,
    shading='auto'
)
ax.set_ylim(0, R_MAX)
ax.set_title('Random-Pulse Radar (CFAR) – Top-Right Quadrant', pad=20)

# Add colorbar for base colormap
sm = plt.cm.ScalarMappable(norm=norm, cmap=base_cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label='Magnitude (arb units)')

# ── Save figure ───────────────────────────────────────────────────────────
def unique_fname(base, ext='png'):
    """Return a filename like base.png, base1.png, etc., that doesn’t yet exist."""
    fname = f"{base}.{ext}"
    i = 1
    while os.path.exists(fname):
        fname = f"{base}{i}.{ext}"
        i += 1
    return fname

outname = unique_fname('radar_image_Cfar_quarter', ext='png')
fig.savefig(outname, dpi=600)
print(f'Saved image as {outname}')
