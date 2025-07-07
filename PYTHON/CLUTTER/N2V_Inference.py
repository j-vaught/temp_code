#!/usr/bin/env python3
import os
import numpy as np
from skimage.io import imread, imsave
from n2v.models import N2V, N2VConfig

# ——— User settings ———
MODEL_DIR    = "/Users/jacobvaught/models/n2v_model"        # where your model folder lives
WEIGHTS_FILE = "/Users/jacobvaught/models/n2v_model/weights_best_copy.h5"       # e.g. the epoch with lowest val_loss
IMAGE_PATH   = "/Users/jacobvaught/Downloads/frames_parallel_9g/frame_01931_t19700101T000001_rc03_uc496_g80.png"    # pick any frame to test
OUT_PATH     = "frame_denoised.png"        # where to write the result

# ——— Helper ———
def read_gray(path):
    img = imread(path)
    if img.ndim == 3:
        if img.shape[2] == 4:    # drop alpha
            img = img[..., :3]
        img = img.mean(axis=2)
    return img.astype(np.float32)

# ——— Load and configure model ———
# We need one sample patch to infer the axes/channel ordering:
sample = read_gray(IMAGE_PATH)[None, ..., None]  # shape (1, Y, X, 1)
config = N2VConfig(sample)
model  = N2V(config, name='n2v_model', basedir=os.path.dirname(MODEL_DIR))

# ——— Load weights ———
model.keras_model.load_weights(os.path.join(os.path.dirname(MODEL_DIR), WEIGHTS_FILE))
print(f"Loaded weights from {WEIGHTS_FILE}")

# ——— Run inference ———
raw = read_gray(IMAGE_PATH)
denoised = model.predict(raw[..., np.newaxis], axes='YXC')

# ——— Save output ———
out_img = np.clip(denoised.squeeze(), 0, 255).astype(np.uint8)
imsave(OUT_PATH, out_img)
print(f"Denoised image written to {OUT_PATH}")
