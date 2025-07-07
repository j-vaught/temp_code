#!/usr/bin/env python3
"""
Noise2Void pipeline with patch-based self-supervised training and manual epoch-level weight saving.
Converts RGBA/3-channel PNGs to grayscale.
Edit constants below: DATA_DIR, MODEL_DIR, hyperparameters, and patch settings.
"""
import os
import numpy as np
import random
from skimage.io import imread, imsave

# ---------- User Config ----------
DATA_DIR = "/Users/jacobvaught/Downloads/frames_parallel_9g"
MODEL_DIR = "/Users/jacobvaught/Downloads/models"
EPOCHS = 50
STEPS = 100            # steps per epoch
BATCH_SIZE = 8         # patches per step
PATCH_SIZE = (64, 64)  # patch height, width
OUTPUT_SUFFIX = '_denoised'

# ---------- Utility Functions ----------
def get_file_list(data_dir):
    return sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().endswith('.png')
    )

def read_gray(path):
    img = imread(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = img.mean(axis=2)
    return img.astype(np.float32)

def extract_random_patches(file_list, patch_size, n_patches):
    ph, pw = patch_size
    patches = np.zeros((n_patches, ph, pw, 1), dtype=np.float32)
    for i in range(n_patches):
        path = random.choice(file_list)
        img = read_gray(path)
        h, w = img.shape
        y = random.randint(0, h - ph)
        x = random.randint(0, w - pw)
        patches[i, ..., 0] = img[y:y+ph, x:x+pw]
    return patches

# ---------- Training Function ----------
def train_noise2void(file_list, model_dir):
    from n2v.models import N2V, N2VConfig

    total_patches = STEPS * BATCH_SIZE
    patches = extract_random_patches(file_list, PATCH_SIZE, total_patches)

    config = N2VConfig(patches)
    model = N2V(config, name='n2v_model', basedir=model_dir)

    # Manual epoch loop for saving weights and sample
    for epoch in range(1, EPOCHS+1):
        print(f"Starting epoch {epoch}/{EPOCHS}")
        # Train one epoch
        model.train(
            patches,
            patches,
            epochs=1,
            steps_per_epoch=STEPS
        )
        # Save weights
        weights_path = os.path.join(model_dir, f"weights_epoch_{epoch:02d}.h5")
        model.keras_model.save_weights(weights_path)
        print(f"Saved weights: {weights_path}")
        # Save sample output
        idx = random.randrange(len(file_list))
        raw = read_gray(file_list[idx])
        pred = model.predict(raw[..., np.newaxis], axes='YXC')
        sample_dir = os.path.join(model_dir, 'samples')
        os.makedirs(sample_dir, exist_ok=True)
        sample_path = os.path.join(sample_dir, f'sample_epoch_{epoch:02d}.png')
        # Convert to uint8 for PNG
        out_img = np.clip(pred.squeeze(), 0, 255).astype(np.uint8)
        imsave(sample_path, out_img)
        print(f"Saved sample: {sample_path}")

    # Final inference
    for path in file_list:
        raw = read_gray(path)
        pred = model.predict(raw[..., np.newaxis], axes='YXC')
        out = path.replace('.png', OUTPUT_SUFFIX + '.png')
        # Convert to uint8 for PNG
        out_img = np.clip(pred.squeeze(), 0, 255).astype(np.uint8)
        imsave(out, out_img)

# ---------- Main ----------
if __name__ == '__main__':
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = get_file_list(DATA_DIR)
    print(f"Found {len(files)} images in {DATA_DIR}")
    train_noise2void(files, MODEL_DIR)
    print("Denoising complete.")
