import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import time

# Function to convert CSV angle to degrees
def convert_angle(angle):
    return angle * (360.0 / 8196.0)

start_time = time.time()

# PRECOMPUTE FOR ALL IMAGES
# Image dimensions
image_size = 1024
center = image_size // 2
radius = center

# Precompute distances
distances = (np.arange(1024) / 1024) * radius

# Precompute trigonometric values
angles = np.linspace(0, 360, 8197)
sin_lookup = np.sin(np.deg2rad(angles))
cos_lookup = np.cos(np.deg2rad(angles))

# Precompute x and y coordinates for all angles
all_x = center + np.outer(distances, cos_lookup).T
all_y = center + np.outer(distances, sin_lookup).T
print(f"Precomputing data: {time.time() - start_time:.2f} seconds")

# Read the CSV file using the C engine
df = pd.read_csv('data/radar0_file46.csv', engine='c')
print(f"Reading CSV: {time.time() - start_time:.2f} seconds")

# Downsample the data: take every nth pulse (n=10)
df = df.iloc[::10, :].reset_index(drop=True)
print(f"Downsampling data: {time.time() - start_time:.2f} seconds")

# Precompute the color flipped echo data, skipping the first column after 'Angle'
echo_data_flipped = 255 - df.iloc[:, 2:].values.astype(int)  # Skip the 'Range' column
print(f"Precomputing color flipped data: {time.time() - start_time:.2f} seconds")

# Create a blank image
image = Image.new('L', (image_size, image_size), 'white')
draw = ImageDraw.Draw(image)
print(f"Creating blank image: {time.time() - start_time:.2f} seconds")

# Draw each pulse
for angle_idx, row in zip(df['Angle'], echo_data_flipped):
    x = all_x[angle_idx]
    y = all_y[angle_idx]
    mask = row != 255  # Create a mask for non-white points
    for xi, yi, color in zip(x[mask], y[mask], row[mask]):
        draw.point((xi, yi), fill=int(color))
print(f"Drawing points: {time.time() - start_time:.2f} seconds")

# Save the image
image.save('radar_echoes.png')
print(f"Saving image: {time.time() - start_time:.2f} seconds")

# Delete the png file after program runs
import os
os.remove('radar_echoes.png')
