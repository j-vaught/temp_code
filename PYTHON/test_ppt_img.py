# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d.art3d import Line3DCollection
# from matplotlib.animation import FuncAnimation, FFMpegWriter

# # --- PARAMETERS ---
# n_gains     = 100              # total dual-pulse sets
# start_z     = 0.01
# end_z       = 10
# frames      = 200              # animation frames
# fps         = 20               # output FPS
# output_file = "radar_zoom_ease_wide.mp4"

# # compute spacing & offset
# gain_spacing = (end_z - start_z) / (n_gains - 1)
# offset       = gain_spacing * 0.1   # 10% of gain spacing

# # z-axis initial window
# small_extent = offset * 5

# # precompute slice depths
# times = np.linspace(start_z, end_z, n_gains)

# # --- SETUP FIGURE (wider for PPT) ---
# fig = plt.figure(figsize=(12, 6))  # wide frame
# ax  = fig.add_subplot(111, projection='3d')
# # emphasize Range/Azimuth plane (wider) with less z stretch
# ax.set_box_aspect([2, 2, 1])
# ax.set_xlabel("Range"); ax.set_ylabel("Azimuth"); ax.set_zlabel("Time")

# # cube wireframe
# pts   = np.array([[x, y, z] for x in (0,1) for y in (0,1) for z in (0,1)])
# edges = [(0,1),(2,3),(4,5),(6,7),
#          (0,2),(1,3),(4,6),(5,7),
#          (0,4),(1,5),(2,6),(3,7)]
# wire  = Line3DCollection([(pts[i], pts[j]) for i,j in edges],
#                          colors='gray', linewidths=1)

# def init():
#     ax.add_collection3d(wire)
#     ax.view_init(elev=20, azim=30)

# def update(frame):
#     ax.cla()
#     ax.add_collection3d(wire)
#     ax.set_xlabel("Range"); ax.set_ylabel("Azimuth"); ax.set_zlabel("Time")
#     ax.view_init(elev=20, azim=30)

#     # ease-in reveal
#     t_frac = frame / (frames - 1)
#     ease   = t_frac**2
#     k      = int(ease * (n_gains - 1)) + 1  # always show at least one slice

#     # plot first k dual-pulse slices with larger planes
#     for t in times[:k]:
#         for j, color in enumerate(('blue', 'orange')):
#             Z = np.full((10, 10), t + j*offset)  # finer grid
#             # expand to full cube footprint
#             X, Y = np.meshgrid(np.linspace(0,1,10), np.linspace(0,1,10))
#             ax.plot_surface(X, Y, Z, color=color, alpha=0.5, shade=False)

#     # zoom-out with same easing
#     z_top = small_extent + (end_z + offset - small_extent) * ease
#     ax.set_zlim(0, z_top)

# # build & save animation
# anim   = FuncAnimation(fig, update, frames=frames, init_func=init, interval=100)
# writer = FFMpegWriter(fps=fps, codec='h264')
# anim.save(output_file, writer=writer)

# print(f"Saved widened animation to {output_file}")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, FFMpegWriter

# --- PARAMETERS ---
grid_size   = 4               # 4×4×4 voxels
mask_ratio  = 0.75            # 75% voxels masked
frames      = 60              # animation frames
fps         = 20              # output FPS
image_file  = "original_block.png"
video_file  = "block_recolor.mp4"

# Generate voxel centers
centers = np.array([
    (x + 0.5, y + 0.5, z + 0.5)
    for x in range(grid_size)
    for y in range(grid_size)
    for z in range(grid_size)
])
N = len(centers)

# Random colors per voxel
np.random.seed(1)
orig_colors = np.random.rand(N, 3)

# Random mask for 75%
np.random.seed(2)
mask = np.random.rand(N) < mask_ratio

# Build unit cube faces
unit = np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],
                 [0,0,1],[1,0,1],[1,1,1],[0,1,1]])
faces = [
    [unit[i] for i in [0,1,2,3]],
    [unit[i] for i in [4,5,6,7]],
    [unit[i] for i in [0,1,5,4]],
    [unit[i] for i in [2,3,7,6]],
    [unit[i] for i in [1,2,6,5]],
    [unit[i] for i in [0,3,7,4]],
]

# Precompute cube meshes for each voxel
voxel_meshes = [np.array(faces) + np.array([cx-0.5, cy-0.5, cz-0.5])
                for (cx, cy, cz) in centers]

# --- STATIC IMAGE: ORIGINAL BLOCK ---
fig = plt.figure(figsize=(6,6))
ax  = fig.add_subplot(111, projection='3d')
ax.set_box_aspect([1,1,1])
ax.axis('off')
ax.set_title("Original Colored Block", pad=10)

# Draw each voxel with its original color
for color, mesh in zip(orig_colors, voxel_meshes):
    col = Poly3DCollection(mesh, facecolors=color, edgecolors='k', linewidths=0.1)
    ax.add_collection3d(col)

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_zlim(0, grid_size)
plt.savefig(image_file, dpi=150)
plt.close()

# --- ANIMATION: MASK → RECOLOR ---
# Initial masked colors
masked_color = np.array([0.7,0.7,0.7])
start_colors  = np.where(mask[:,None], masked_color, orig_colors)

# Setup figure
fig2 = plt.figure(figsize=(6,6))
ax2  = fig2.add_subplot(111, projection='3d')
ax2.set_box_aspect([1,1,1])
ax2.axis('off')
ax2.set_title("MAE Recolor Animation", pad=10)

# Create Poly3DCollections for each voxel
collections = []
for i, mesh in enumerate(voxel_meshes):
    col = Poly3DCollection(mesh, facecolors=start_colors[i], edgecolors='k', linewidths=0.1)
    ax2.add_collection3d(col)
    collections.append(col)

ax2.set_xlim(0, grid_size)
ax2.set_ylim(0, grid_size)
ax2.set_zlim(0, grid_size)

def update(frame):
    t = frame / (frames - 1)
    for i, col in enumerate(collections):
        if mask[i]:
            # fade from gray to original
            c = (1 - t) * masked_color + t * orig_colors[i]
        else:
            c = orig_colors[i]
        col.set_facecolor(c)
    return collections

anim = FuncAnimation(fig2, update, frames=frames, blit=False, interval=100)

# Save animation
writer = FFMpegWriter(fps=fps, codec='h264')
anim.save(video_file, writer=writer)
plt.close(fig2)

print(f"Saved image to {image_file}")
print(f"Saved animation to {video_file}")
