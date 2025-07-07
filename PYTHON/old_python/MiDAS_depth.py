import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def debug_midas_pipeline(image_path):
    # 1) Load image
    img = cv2.imread(str(image_path))
    print('1) Loaded image from', image_path)
    print('   - Original image shape (H, W, C):', img.shape)
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print('2) Converted to RGB')
    print('   - RGB image shape:', img_rgb.shape)
    
    # 2) Load MiDaS model and transforms
    print('3) Loading MiDaS model...')
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
    model.eval()
    print('   - MiDaS model loaded:', model.__class__.__name__)
    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.dpt_transform
    print('   - Transform function:', transform)
    
    # 3) Apply transform (pass the raw array)
    transformed = transform(img_rgb)
    print('4) Applied transform:')
    # Handle both dict and tensor return types:
    if isinstance(transformed, dict):
        inp = transformed["image"]
        print('   - Extracted "image" from dict, shape:', inp.shape)
    else:
        inp = transformed
        print('   - Using tensor directly, shape:', inp.shape)
    # Ensure we have exactly [batch, channels, H, W]
    if inp.ndim == 3:
        inp = inp.unsqueeze(0)
        print('   - Added batch dim, new shape:', inp.shape)
    elif inp.ndim == 4:
        print('   - Already batched, shape remains:', inp.shape)
    else:
        raise ValueError(f"Unexpected tensor rank {inp.ndim}; cannot batchify")
    
    # 4) Forward pass
    print('5) Running forward pass...')
    with torch.no_grad():
        pred = model(inp)
    print('   - Raw model output shape:', pred.shape)
    
    # 5) Upsample to original size
    print('6) Upsampling depth to original image size...')
    pred_up = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=img_rgb.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()
    print('   - Upsampled depth shape:', pred_up.shape)
    
    depth = pred_up.cpu().numpy()
    print('   - Depth array dtype:', depth.dtype)
    print('   - Depth min, max:', depth.min(), depth.max())
    
    return img, depth  # return both original BGR image and depth map

def save_depth_overlay(img_bgr, depth_map, out_path, alpha=0.5, cmap=plt.cm.jet):
    # normalize depth to [0,1]
    dm_norm = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    # apply colormap (RGB in range 0–1)
    dm_color = cmap(dm_norm)[:, :, :3]
    dm_color = (dm_color * 255).astype(np.uint8)
    # convert to BGR for OpenCV
    dm_bgr = cv2.cvtColor(dm_color, cv2.COLOR_RGB2BGR)
    # blend
    overlay = cv2.addWeighted(img_bgr, 1 - alpha, dm_bgr, alpha, 0)
    # write out
    cv2.imwrite(str(out_path), overlay)
    print(f"Saved depth overlay to {out_path}")

# if __name__ == "__main__":
#     input_path = Path("overlay_mixed.png")
#     img, depth_map = debug_midas_pipeline(input_path)
#     stem, ext = input_path.stem, input_path.suffix
#     output_path = input_path.parent / f"{stem}_depth{ext}"
#     save_depth_overlay(img, depth_map, output_path)


import cv2
import numpy as np

def orthorectify(img_bgr, depth_map, K, R=None, t=None):
    h, w = depth_map.shape
    # 1) back‐project
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    Z = depth_map
    X = (u - K[0,2]) * Z / K[0,0]
    Y = (v - K[1,2]) * Z / K[1,1]

    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)  # (N,3)

    # 2) optionally transform to a world frame
    if R is not None and t is not None:
        pts = (R @ pts.T + t).T

    Xo, Yo = pts[:,0], pts[:,1]

    # 3) normalize to pixel‐coords
    Xo = (Xo - Xo.min()) / (Xo.max() - Xo.min()) * (w - 1)
    Yo = (Yo - Yo.min()) / (Yo.max() - Yo.min()) * (h - 1)

    map_x = Xo.reshape(h, w).astype(np.float32)
    map_y = Yo.reshape(h, w).astype(np.float32)

    # 4) remap
    ortho = cv2.remap(img_bgr, map_x, map_y,
                      interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT)
    return ortho

if __name__ == "__main__":
    # --- load your depth and image ---
    img_bgr, depth_map = debug_midas_pipeline("overlay_mixed.png")

    # --- define or approximate intrinsics ---
    h, w = depth_map.shape
    fx = fy = 0.5 * max(h, w)
    cx, cy = w/2, h/2
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]])

    # --- (optional) if you have camera pose R, t leave them here, else pass R=None, t=None ---
    ortho = orthorectify(img_bgr, depth_map, K)

    # --- save result ---
    cv2.imwrite("orthorectified.png", ortho)
    print("Saved orthorectified image as orthorectified.png")
