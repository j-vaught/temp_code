#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Rotating‐radar magnitude simulator with:
 • Square & circle targets (hard/soft edges + roughness)
 • Rain attenuation + rain clutter
 • Sea / wave clutter
 • Range‐ and azimuth‐domain smoothing
 • Polar PPI display
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. Radar & display parameters ─────────────────────────────── EDIT HERE
# ---------------------------------------------------------------
R_MAX          = 1024         # metres
N_RANGE_BINS   = 1024         # range cells
N_AZIM_BINS    = 3600          # 1° resolution
BEAM_HPBW_DEG  = 1.8          # half‐power beam‐width (°)

RANGE_KERNEL   = [1/3, 1/3, 1/3]  # range blur
AZ_KERNEL_SPAN = 3                # ±3° beam blur
COLORMAP       = "viridis"

c_min = 0.0
c_max = 1.0 # this is how i artifically 'control' gain. i just lower the trheshold/max and it causes the image to be more saturated
levels = 16 # quantization levels for display (simrad = 16; furuno = 256)
P_max = 1.0  # maximum value in P (after smoothing)


ENABLE_RAIN      = False
ENABLE_SEA       = True
BASE_RAIN_MM_HR  = 10.0
RAIN_K           = 0.0007
RAIN_BETA        = 0.93
RAIN_BACKSCALER  = 5e-4
RAIN_RATE_FIELD  = None

SEA_SIGMA0_MEAN = 0.2 # this is the noise in each image. it is random for each image
SEA_SIGMA0_STD  = 0.4 # this is the noise in each image. it is random for each image


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


# ---------------------------------------------------------------
# 3. Rain & sea clutter ──────────────────────────────────────── EDIT HERE
# ---------------------------------------------------------------

def sea_mask(x,y): return y < 0    # y<0 is sea

# ---------------------------------------------------------------
# 4. Precompute grids & noise profiles
# ---------------------------------------------------------------
rng_res     = R_MAX / N_RANGE_BINS
az_res_rad  = 2*np.pi / N_AZIM_BINS
rng_centres = np.linspace(rng_res/2, R_MAX - rng_res/2, N_RANGE_BINS)
az_centres  = np.linspace(0, 2*np.pi - az_res_rad, N_AZIM_BINS)

# Prepare an empty P
P = np.zeros((N_AZIM_BINS, N_RANGE_BINS), float)

# Precompute a per‐azimuth noise profile for rough edges:
for tgt in TARGETS:
    rough = tgt.get("roughness", 0.0)
    tgt["noise_profile"] = np.random.uniform(
        low=-rough, high=rough, size=N_AZIM_BINS
    ) if rough > 0 else np.zeros(N_AZIM_BINS)

# ---------------------------------------------------------------
# 5. Rasterize targets into P[az,range]
# ---------------------------------------------------------------
for iaz, th in enumerate(az_centres):
    cos_t, sin_t = np.cos(th), np.sin(th)
    xs = rng_centres * cos_t
    ys = rng_centres * sin_t

    for tgt in TARGETS:
        # fetch noise offset
        edge_noise = tgt["noise_profile"][iaz]

        # cartesian offset to center
        dx = xs - tgt["cx"]
        dy = ys - tgt["cy"]

        if tgt["type"] == "square":
            eff_half = tgt["half"] + edge_noise
            inside = (np.abs(dx) <= eff_half) & (np.abs(dy) <= eff_half)
            if tgt["ramp"] > 0:
                d_edge = np.minimum(eff_half - np.abs(dx),
                                    eff_half - np.abs(dy))
        elif tgt["type"] == "circle":
            r       = np.hypot(dx, dy)
            eff_rad = tgt["radius"] + edge_noise
            inside  = r <= eff_rad
            if tgt["ramp"] > 0:
                d_edge = eff_rad - r
        else:
            raise ValueError("Unknown target type: "+tgt["type"])

        # amplitude: linear ramp or hard
        if tgt["ramp"] > 0:
            amp = np.clip(d_edge / tgt["ramp"], 0.0, 1.0)
        else:
            amp = 1.0

        # accumulate
        P[iaz] += inside * amp

# ---------------------------------------------------------------
# 6. Rain attenuation & clutter (unchanged) ─────────────────── see above
# ---------------------------------------------------------------
if ENABLE_RAIN:
    # ... same as before, filling R, computing L_rain, etc.
    # P *= L_rain; P += RAIN_BACKSCALER*R
    pass

# ---------------------------------------------------------------
# 7. Sea clutter ─────────────────────────────────────────────── see above
# ---------------------------------------------------------------
if ENABLE_SEA:
    AZ, RNG = np.meshgrid(az_centres, rng_centres, indexing="ij")
    X, Y    = RNG*np.cos(AZ), RNG*np.sin(AZ)
    mask    = sea_mask(X, Y)
    clutter = SEA_SIGMA0_MEAN + SEA_SIGMA0_STD*np.random.rayleigh(size=P.shape)

    # add clutter and immediately cap at P_max
    P += mask * clutter
    np.clip(P, None, P_max, out=P)

# ---------------------------------------------------------------
# 8. Range & azimuth smoothing ───────────────────────────────── see above
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
P_smooth  = sum(w * np.roll(P, k, axis=0) for k,w in zip(offs,akw))

# ---------------------------------------------------------------
# 8b. Clamp everything to [0,1]
# ---------------------------------------------------------------
P_smooth = np.clip(P_smooth, c_min, c_max)

# ---------------------------------------------------------------
# 8c. Quantize to 16 levels between c_min and c_max
# ---------------------------------------------------------------
P_norm = (P_smooth - c_min) / (c_max - c_min)
P_idx = np.floor(P_norm * levels).astype(int)
P_idx[P_idx == levels] = levels - 1
bin_edges = np.linspace(c_min, c_max, levels + 1)
bin_mids  = 0.5 * (bin_edges[:-1] + bin_edges[1:])
P_quant   = bin_mids[P_idx]

# ---------------------------------------------------------------
# 9. Plot polar PPI with quantized data
# ---------------------------------------------------------------
theta_edges = np.linspace(0, 2*np.pi, N_AZIM_BINS+1)
rng_edges   = np.linspace(0, R_MAX, N_RANGE_BINS+1)

fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(8,8))
pcm = ax.pcolormesh(
    theta_edges, rng_edges, P_quant.T,
    shading="auto",
    cmap=COLORMAP,
    vmin=c_min,
    vmax=c_max
)
ax.set_ylim(0, R_MAX)
ax.set_title("SIMRAD Simulator", pad=20)
fig.colorbar(pcm, ax=ax, label="Magnitude (arb units)")
# plt.show()





import os

def unique_fname(base, ext="png"):
    """Return a filename like base.png, base1.png, base2.png… that doesn’t yet exist."""
    fname = f"{base}.{ext}"
    i = 0
    while os.path.exists(fname):
        fname = f"{base}{i}.{ext}"
        i += 1
    return fname

# … after creating your figure `fig` …
fname = unique_fname("radar_image_clutter", ext="png")
fig.savefig(fname, dpi=600)
print(f"Saved image as {fname}")
