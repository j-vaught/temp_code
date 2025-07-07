#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rotating‐radar magnitude simulator with gain‐sweep histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import math

# ---------------------------------------------------------------
# 1. Radar & display parameters ─────────────────────────────── EDIT HERE
# ---------------------------------------------------------------
R_MAX          = 1024         # metres
N_RANGE_BINS   = 1024         # range cells
N_AZIM_BINS    = 3600         # 1° resolution
BEAM_HPBW_DEG  = 1.8          # half‐power beam‐width (°)

RANGE_KERNEL   = [1/3, 1/3, 1/3]  # range blur
AZ_KERNEL_SPAN = 3                # ±3° beam blur
COLORMAP       = "viridis"

c_min = 0.0
# list of gains (C_max) to sweep through:
C_MAX_LIST = [ 1, 5, 10, 15, 20 ]  # e.g., [1, 5, 10, 15, 20]
# , 5, 10, 15, 20

levels = 16      # quantization levels for display
P_max = 20.0      # maximum before smoothing

ENABLE_RAIN      = False
ENABLE_SEA       = True
BASE_RAIN_MM_HR  = 10.0
RAIN_K           = 0.0007
RAIN_BETA        = 0.93
RAIN_BACKSCALER  = 5e-4

SEA_SIGMA0_MEAN = 0.2
SEA_SIGMA0_STD  = 0.4

# ---------------------------------------------------------------
# 2. Targets (square or circle) ─────────────────────────────── EDIT HERE
# ---------------------------------------------------------------
# type: "square" or "circle"
# squares use "half", circles use "radius"
# both accept "ramp" (m) and "roughness" (m)
TARGETS = [
    # original square
    {
      "type":      "square",
      "cx":         500.0,
      "cy":        -500.0,
      "half":       15.0,
      "ramp":       10.0,
      "roughness":  3.0
    },
    # soft‐edge, jagged circle of radius 40 m
    {
      "type":      "circle",
      "cx":         200.0,
      "cy":        -800.0,
      "radius":     40.0,
      "ramp":       10.0,
      "roughness":  5.0
    },
    # mirrored into negative X
    {
      "type":      "circle",
      "cx":        -200.0,
      "cy":        -800.0,
      "radius":     12.0,
      "ramp":       10.0,
      "roughness":  5.0
    },
    # up-near corner, negative X
    {
      "type":      "circle",
      "cx":         -50.0,
      "cy":         -50.0,
      "radius":     10.0,
      "ramp":        5.0,
      "roughness":   1.5
    },
    # original edge
    {
      "type":      "square",
      "cx":         950.0,
      "cy":        -100.0,
      "half":        8.0,
      "ramp":        4.0,
      "roughness":   2.0
    },
    # deep-water marker, negative X
    {
      "type":      "circle",
      "cx":        -100.0,
      "cy":        -900.0,
      "radius":     12.0,
      "ramp":        4.0,
      "roughness":   2.0
    },
    # adjusted to just inside 1024-unit radius
    {
      "type":      "square",
      "cx":         700.0,
      "cy":        -730.0,
      "half":       10.0,
      "ramp":        5.0,
      "roughness":   3.0
    },
    # mid-channel
    {
      "type":      "circle",
      "cx":         500.0,
      "cy":         -50.0,
      "radius":     15.0,
      "ramp":        5.0,
      "roughness":   2.0
    },
    # western flank, negative X
    {
      "type":      "square",
      "cx":         -50.0,
      "cy":        -500.0,
      "half":       14.0,
      "ramp":        5.0,
      "roughness":   2.0
    },
    # adjusted to just inside 1024-unit radius
    {
      "type":      "circle",
      "cx":         850.0,
      "cy":        -477.0,
      "radius":      9.0,
      "ramp":        4.0,
      "roughness":   1.0
    },
    # adjusted to just inside 1024-unit radius
    {
      "type":      "square",
      "cx":         400.0,
      "cy":        -906.0,
      "half":       11.0,
      "ramp":        3.0,
      "roughness":   1.5
    }
]

def sea_mask(x,y): return y < 0    # y<0 is sea

# ---------------------------------------------------------------
# 3. Precompute grids & noise profiles
# ---------------------------------------------------------------
rng_res     = R_MAX / N_RANGE_BINS
az_res_rad  = 2*np.pi / N_AZIM_BINS
rng_centres = np.linspace(rng_res/2, R_MAX - rng_res/2, N_RANGE_BINS)
az_centres  = np.linspace(0, 2*np.pi - az_res_rad, N_AZIM_BINS)

# grid of X,Y for masking by y>0
AZ, RNG = np.meshgrid(az_centres, rng_centres, indexing="ij")
X = RNG * np.cos(AZ)
Y = RNG * np.sin(AZ)

# Prepare an empty P
P = np.zeros((N_AZIM_BINS, N_RANGE_BINS), float)

# Precompute per‐azimuth noise for rough edges
for tgt in TARGETS:
    rough = tgt.get("roughness", 0.0)
    tgt["noise_profile"] = (
        np.random.uniform(-rough, rough, N_AZIM_BINS)
        if rough>0 else np.zeros(N_AZIM_BINS)
    )

# ---------------------------------------------------------------
# 4. Rasterize targets into P[az,range]
# ---------------------------------------------------------------
for iaz, th in enumerate(az_centres):
    cos_t, sin_t = np.cos(th), np.sin(th)
    xs = rng_centres * cos_t
    ys = rng_centres * sin_t

    for tgt in TARGETS:
        edge_noise = tgt["noise_profile"][iaz]
        dx = xs - tgt["cx"]
        dy = ys - tgt["cy"]

        if tgt["type"] == "square":
            eff_half = tgt["half"] + edge_noise
            inside   = (np.abs(dx)<=eff_half) & (np.abs(dy)<=eff_half)
            if tgt["ramp"]>0:
                d_edge = np.minimum(eff_half-np.abs(dx),
                                    eff_half-np.abs(dy))
        elif tgt["type"] == "circle":
            r       = np.hypot(dx, dy)
            eff_rad = tgt["radius"] + edge_noise
            inside  = r <= eff_rad
            if tgt["ramp"]>0:
                d_edge = eff_rad - r
        else:
            raise ValueError("Unknown target type: "+tgt["type"])

        amp = np.clip(d_edge/tgt["ramp"],0,1) if tgt["ramp"]>0 else 1.0
        P[iaz] += inside * amp

# ---------------------------------------------------------------
# 5. Rain & sea clutter ────────────────────────────────────────
# ---------------------------------------------------------------
if ENABLE_RAIN:
    # … your rain code …
    pass

if ENABLE_SEA:
    clutter = SEA_SIGMA0_MEAN + SEA_SIGMA0_STD*np.random.rayleigh(size=P.shape)
    mask_sea = sea_mask(X, Y)
    P += mask_sea * clutter
    np.clip(P, None, P_max, out=P)

# ---------------------------------------------------------------
# 6. Range & azimuth smoothing ─────────────────────────────────
# ---------------------------------------------------------------
# range blur
rk = np.asarray(RANGE_KERNEL)
rk /= rk.sum()
P = np.apply_along_axis(lambda v: np.convolve(v, rk, mode="same"),
                        axis=1, arr=P)

# azimuth blur
sigma_deg = BEAM_HPBW_DEG/(2*np.sqrt(2*np.log(2)))
offs      = np.arange(-AZ_KERNEL_SPAN, AZ_KERNEL_SPAN+1)
akw       = np.exp(-0.5*(offs/sigma_deg)**2)
akw      /= akw.sum()
P_smooth_full = sum(w * np.roll(P, k, axis=0)
                    for k,w in zip(offs,akw))

# quantization bins (shared across all gains)
bin_edges = np.linspace(c_min, max(C_MAX_LIST), levels+1)
bin_mids  = 0.5*(bin_edges[:-1] + bin_edges[1:])

# ---------------------------------------------------------------
# 7. Loop over each gain, clamp+quantize, then histogram in subplots ─
# ---------------------------------------------------------------
n_gains = len(C_MAX_LIST)
# choose rows and cols for subplots: e.g., as square as possible
n_cols = int(math.ceil(math.sqrt(n_gains)))
n_rows = int(math.ceil(n_gains / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, 
                         figsize=(4*n_cols, 3*n_rows), 
                         sharex=True, sharey=True)

# flatten in case axes is 2D or 1D
axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]

for ax, C_max in zip(axes_flat, C_MAX_LIST):
    # clamp & normalize
    P_clamped = np.clip(P_smooth_full, c_min, C_max)
    P_norm    = (P_clamped - c_min) / (C_max - c_min)
    idx       = np.floor(P_norm * levels).astype(int)
    idx[idx==levels] = levels-1
    P_quant   = bin_mids[idx]

    # mask out positive‐y (ignore y>0)
    data = P_quant[Y<0].ravel()

    # plot histogram
    ax.hist(data, bins=bin_edges, density=True)
    ax.set_title(f'C_max = {C_max}')
    ax.grid(True)

# turn off any unused axes
for ax in axes_flat[n_gains:]:
    ax.axis('off')

fig.text(0.5, 0.04, 'Quantized Magnitude', ha='center')
fig.text(0.04, 0.5, 'Probability Density', va='center', rotation='vertical')
fig.suptitle('Histograms of Plotted Data (y<0) for Each Gain', y=0.95)
plt.tight_layout(rect=[0.04, 0.04, 1, 0.94])
plt.show()

