#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-gain radar simulator + ML-based denoising via Ridge regression

Produces:
  - radar_gain_<gain>.png       (one per gain, each with fresh clutter)
  - radar_denoised_ml.png       (Ridge-based fusion)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------
# 1. Radar & display parameters
# ---------------------------------------------------------------
R_MAX          = 1024           # metres
N_RANGE_BINS   = 1024           # range cells
N_AZIM_BINS    = 3600           # 1° resolution
BEAM_HPBW_DEG  = 1.8            # half-power beam-width (°)

RANGE_KERNEL   = [1/3, 1/3, 1/3]  # range blur
AZ_KERNEL_SPAN = 3                # ±3° beam blur
LEVELS         = 16               # quantization levels
P_max          = 2.0              # cap after clutter/smoothing

ENABLE_RAIN      = False
ENABLE_SEA       = True
SEA_SIGMA0_MEAN = 0.3             # sea clutter mean
SEA_SIGMA0_STD  = 0.6             # sea clutter randomness
scaling = 0.6*0.8*0.9 # scaling factor for ML denoising to rescale the max values where they should be
GAINS = [0.5, 0.75, 1.0, 1.25]

# ---------------------------------------------------------------
# 2. Targets (square or circle)
# ---------------------------------------------------------------
TARGETS = [
    { "type":"square", "cx":500.0,  "cy":-500.0, "half":15.0, "ramp":10.0, "roughness":3.0 },
    { "type":"circle", "cx":200.0,  "cy":-800.0, "radius":40.0, "ramp":10.0, "roughness":5.0 },
    { "type":"circle", "cx":-200.0, "cy":-800.0, "radius":12.0, "ramp":10.0, "roughness":5.0 },
    { "type":"circle", "cx":-50.0,  "cy":-50.0,  "radius":10.0, "ramp":5.0,  "roughness":1.5 },
    { "type":"square", "cx":950.0,  "cy":-100.0, "half":8.0,  "ramp":4.0,  "roughness":2.0 },
    { "type":"circle", "cx":-100.0, "cy":-900.0, "radius":12.0, "ramp":4.0,  "roughness":2.0 },
    { "type":"square", "cx":700.0,  "cy":-730.0, "half":10.0, "ramp":5.0,  "roughness":3.0 },
    { "type":"circle", "cx":500.0,  "cy":-50.0,  "radius":15.0, "ramp":5.0,  "roughness":2.0 },
    { "type":"square", "cx":-50.0,  "cy":-500.0, "half":14.0, "ramp":5.0,  "roughness":2.0 },
    { "type":"circle", "cx":850.0,  "cy":-477.0, "radius":9.0,  "ramp":4.0,  "roughness":1.0 },
    { "type":"square", "cx":400.0,  "cy":-906.0, "half":11.0, "ramp":3.0,  "roughness":1.5 }
]

def sea_mask(x, y):
    return y < 0  # sea where y<0

# ---------------------------------------------------------------
# 3. Filenames helper
# ---------------------------------------------------------------
def unique_fname(base, ext="png"):
    fname = f"{base}.{ext}"
    i = 1
    while os.path.exists(fname):
        fname = f"{base}{i}.{ext}"
        i += 1
    return fname

# ---------------------------------------------------------------
# 4. Precompute grids & target-only field
# ---------------------------------------------------------------
rng_res     = R_MAX / N_RANGE_BINS
az_res_rad  = 2*np.pi / N_AZIM_BINS
rng_centres = np.linspace(rng_res/2, R_MAX - rng_res/2, N_RANGE_BINS)
az_centres  = np.linspace(0, 2*np.pi - az_res_rad, N_AZIM_BINS)

# build P_targets once
P_targets = np.zeros((N_AZIM_BINS, N_RANGE_BINS), float)
for tgt in TARGETS:
    rough = tgt.get("roughness", 0.0)
    tgt["noise_profile"] = (np.random.uniform(-rough, rough, N_AZIM_BINS)
                             if rough > 0 else np.zeros(N_AZIM_BINS))
for iaz, th in enumerate(az_centres):
    ct, st = np.cos(th), np.sin(th)
    xs, ys = rng_centres * ct, rng_centres * st
    for tgt in TARGETS:
        noise = tgt["noise_profile"][iaz]
        dx, dy = xs - tgt["cx"], ys - tgt["cy"]
        if tgt["type"] == "square":
            h = tgt["half"] + noise
            inside = (np.abs(dx)<=h)&(np.abs(dy)<=h)
            if tgt["ramp"]>0:
                d_edge = np.minimum(h-np.abs(dx), h-np.abs(dy))
        else:  # circle
            r0  = np.hypot(dx,dy)
            rad = tgt["radius"] + noise
            inside = (r0<=rad)
            if tgt["ramp"]>0:
                d_edge = rad - r0
        amp = (np.clip(d_edge/tgt["ramp"], 0,1)
               if tgt["ramp"]>0 else 1.0)
        P_targets[iaz] += inside * amp

# ---------------------------------------------------------------
# 5. Smoothing function
# ---------------------------------------------------------------
def smooth_field(P):
    # range blur
    rk = np.array(RANGE_KERNEL); rk/=rk.sum()
    Pr = np.apply_along_axis(lambda v: np.convolve(v, rk, mode="same"),
                             axis=1, arr=P)
    # azimuth blur
    sigma = BEAM_HPBW_DEG/(2*np.sqrt(2*np.log(2)))
    offs  = np.arange(-AZ_KERNEL_SPAN, AZ_KERNEL_SPAN+1)
    akw   = np.exp(-0.5*(offs/sigma)**2)
    akw  /= akw.sum()
    Pa    = sum(w*np.roll(Pr, k, axis=0) for k,w in zip(offs,akw))
    return np.clip(Pa, 0.0, 1.0)

# precompute clean smooth field
P_clean = smooth_field(P_targets)


# ---------------------------------------------------------------
# 4.5. Ground-truth PPI (no clutter)
# ---------------------------------------------------------------
fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(6,6))
pcm = ax.pcolormesh(
    np.linspace(0, 2*np.pi, N_AZIM_BINS+1),
    np.linspace(0, R_MAX,   N_RANGE_BINS+1),
    P_clean.T, shading="auto"
)
ax.set_ylim(0, R_MAX)
ax.set_title("Ground Truth (No Clutter)", pad=20)
fig.colorbar(pcm, ax=ax, label="Mag (arb)")
fname_gt = unique_fname("radar_ground_truth")
fig.savefig(fname_gt, dpi=300)
plt.close(fig)
print("Saved", fname_gt)


# prepare mesh for clutter
AZ, RNG = np.meshgrid(az_centres, rng_centres, indexing="ij")
X, Y    = RNG*np.cos(AZ), RNG*np.sin(AZ)

# ---------------------------------------------------------------
# 6. Multi-gain capture with fresh clutter each time
# ---------------------------------------------------------------
stack = []
for g in GAINS:
    # re-simulate sea clutter for this gain-iteration
    if ENABLE_SEA:
        clutter = (SEA_SIGMA0_MEAN
                   + SEA_SIGMA0_STD*np.random.rayleigh(size=P_targets.shape))
        P_noisy_raw = P_targets + sea_mask(X,Y)*clutter
    else:
        P_noisy_raw = P_targets.copy()
    P_noisy_raw = np.clip(P_noisy_raw, None, P_max)

    # smooth this noisy realization
    P_noisy_smooth = smooth_field(P_noisy_raw)

    # quantize at this gain
    Pg  = np.clip(P_noisy_smooth * g, 0.0, 1.0)
    idx = np.floor(Pg * LEVELS).astype(int)
    idx[idx==LEVELS] = LEVELS-1
    edges = np.linspace(0,1,LEVELS+1)
    mids  = 0.5*(edges[:-1] + edges[1:])
    Pq    = mids[idx]
    stack.append(Pq)

    # save PPI
    fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(6,6))
    pcm = ax.pcolormesh(
        np.linspace(0,2*np.pi,N_AZIM_BINS+1),
        np.linspace(0,R_MAX,  N_RANGE_BINS+1),
        Pq.T, shading="auto"
    )
    ax.set_ylim(0, R_MAX)
    ax.set_title(f"Gain = {g:.1f}", pad=20)
    fig.colorbar(pcm, ax=ax, label="Mag (arb)")
    fname = unique_fname(f"radar_gain_{g:.1f}")
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print("Saved", fname)


# ---------------------------------------------------------------
# 8. ML denoising: train Ridge to map stack→clean
# ---------------------------------------------------------------
# Prepare data
X = np.stack([arr.flatten() for arr in stack], axis=1)  # shape (n_pixels, n_gains)
y = P_clean.flatten()                                   # shape (n_pixels,)

# train
model = Ridge(alpha=1.0)
model.fit(X, y)

# predict & reshape
y_pred = model.predict(X).reshape(P_clean.shape)

# scale by 40 and then clip back to [0,1]
y_pred = np.clip(y_pred * 60/scaling, 0.0, 1.0)


# save ML-denoised result
fig, ax = plt.subplots(subplot_kw={"projection":"polar"}, figsize=(6,6))
ax.pcolormesh(np.linspace(0,2*np.pi,N_AZIM_BINS+1),
              np.linspace(0,R_MAX,  N_RANGE_BINS+1),
              y_pred.T, shading="auto")
ax.set_ylim(0, R_MAX)
ax.set_title("Ridge-stack Denoised", pad=20)
fig.colorbar(ax.collections[0], ax=ax, label="Mag (arb)")
fname_ml = unique_fname("radar_denoised_ml")
fig.savefig(fname_ml, dpi=300)
plt.close(fig)
print("Saved", fname_ml)