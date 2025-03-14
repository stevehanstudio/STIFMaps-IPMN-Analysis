from STIFMaps import STIFMap_generation
from STIFMaps.misc import get_step

import os
import re
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy import interpolate
from PIL import Image
import tifffile
import time

# Directories
project_dir = '/home/steve/Projects/WeaverLab/STIFMaps'
models_dir = '/home/steve/Projects/WeaverLab/STIFMap_dataset/trained_models'
IPMN_directory = os.path.join(project_dir, "IPMN_images/normalized_tiled_v2")
STIFMaps_directory = os.path.join(project_dir, "STIFMap_normalized_images_v2")
os.makedirs(STIFMaps_directory, exist_ok=True)

# Define the base name and the number of rows and columns
base_name = "27620"
base_name_C0 = "27620_C0_full_tile"
base_name_C1 = "27620_C1_full_tile"

# STIFMap models
models = [
    os.path.join(models_dir, 'iteration_1171.pt'),
    os.path.join(models_dir, 'iteration_1000.pt'),
    os.path.join(models_dir, 'iteration_1043.pt'),
    os.path.join(models_dir, 'iteration_1161.pt'),
    os.path.join(models_dir, 'iteration_1180.pt')
]

# Parameters
scale_factor = 2.712
step = get_step(40, scale_factor)
square_side = get_step(224, scale_factor)
batch_size = 100

print('Step size is ' + str(step) + ' pixels')
print('Side length for a square is ' + str(square_side) + ' pixels')

# Function to check image dimensions
def check_image_dimensions(image_path):
    try:
        with tifffile.TiffFile(image_path) as tif:
            width, height = tif.pages[0].shape[:2]
            print(f"File: {os.path.basename(image_path)}, Dimensions: {width}x{height}")
            return width, height
    except Exception as e:
        print(f"Error opening {image_path}: {e}")

# Function to stitch images together
def stitch_images(output_filename, base_name, rows, cols, image_format='png'):
    sample_image_path = f"{base_name}_{0}_{0}.{image_format}"
    sample_image = Image.open(sample_image_path)
    image_width, image_height = sample_image.size

    stitched_width = cols * image_width
    stitched_height = rows * image_height
    stitched_image = Image.new('RGB', (stitched_width, stitched_height))

    for row in range(rows):
        for col in range(cols):
            image_filename = f"{base_name}_{row}_{col}.{image_format}"
            image = Image.open(image_filename)
            x = col * image_width
            y = row * image_height
            stitched_image.paste(image, (x, y))

    stitched_image.save(output_filename)
    print(f"Stitched image saved as {output_filename}")

# Function to convert seconds to hours, minutes, and seconds
def convert_seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours} hours")
    if minutes > 0:
        time_parts.append(f"{minutes} minutes")
    time_parts.append(f"{seconds:.1f} seconds")
    return ", ".join(time_parts)

# Function to generate the output path for STIFMap images
def gen_output_path(filename):
    pattern = r"_tile_(\d+)_(\d+)\.tif$"
    match = re.search(pattern, filename)
    if match:
        row = int(match.group(1))
        col = int(match.group(2))
    else:
        print("Error: row or column name not found in tiled C0 or C1 file name.")
        return None

    output_file = f"{base_name}_STIFMap_{row}_{col}.png"
    return os.path.join(STIFMaps_directory, output_file)

# Function to generate and save the STIFMap
def run_STIFMap(dapi, collagen, name, step, models, batch_size, square_side, check_existing=True):
    output_path = gen_output_path(dapi)

    # Check if the tile has already been processed
    if check_existing and os.path.exists(output_path):
        print(f"Skipping already processed tile: {dapi} and {collagen}")
        return

    start_time = time.perf_counter()

    z_out = STIFMap_generation.generate_STIFMap(dapi, collagen, name, step, models=models,
                                               mask=False, batch_size=batch_size, square_side=square_side,
                                               save_dir=False)

    end_time = time.perf_counter()
    print("Elapsed time:", convert_seconds_to_hms(end_time - start_time))

    output_image = np.mean(z_out, axis=0)

    global_min = np.min(output_image)
    global_max = np.max(output_image)
    output_image_normalized = (output_image - global_min) / (global_max - global_min)

    plt.imsave(output_path, output_image_normalized, cmap="viridis")
    print(f"Saved image: {output_path}")

    # Save the raw stiffness values as a NumPy array
    stiffness_values_path = output_path.replace(".png", ".npy")
    np.save(stiffness_values_path, output_image)
    print(f"Saved stiffness values: {stiffness_values_path}")

# Function to check if a tile has already been processed
def is_tile_completed(output_path):
    return os.path.exists(output_path)

# Automatically determine num_rows and num_cols based on filenames
tile_pattern = re.compile(rf"{base_name_C0}_(\d+)_(\d+)\.tif")

# Dictionary to store how many columns exist per row
row_col_map = {}

for file in os.listdir(IPMN_directory):
    match = tile_pattern.match(file)
    if match:
        row, col = map(int, match.groups())
        row_col_map.setdefault(row, set()).add(col)

# Determine the number of rows and columns
max_row = max(row_col_map.keys())
max_col = max(max(cols) for cols in row_col_map.values())
num_rows = max_row + 1
num_cols = max_col + 1

print(f"Detected grid size: {num_rows} rows x {num_cols} columns")

# Create a list of file names for C0 and C1 tiles
C0_files = [f"{base_name_C0}_{i}_{j}.tif" for i in range(num_rows) for j in range(num_cols)]
C1_files = [f"{base_name_C1}_{i}_{j}.tif" for i in range(num_rows) for j in range(num_cols)]

# Main: Loop through all the tiled C0 and C1 images and pass each one to run_STIFMap()
for row in range(num_rows):  # Iterate over all rows
    for col in range(num_cols):
        dapi_path = os.path.join(IPMN_directory, f"{base_name_C0}_{row}_{col}.tif")
        collagen_path = os.path.join(IPMN_directory, f"{base_name_C1}_{row}_{col}.tif")

        output_path = gen_output_path(dapi_path)

        # Ensure files exist before processing
        if not os.path.exists(dapi_path) or not os.path.exists(collagen_path):
            print(f"Skipping missing tile: {dapi_path} or {collagen_path}")
            continue

        if is_tile_completed(output_path):
            print(f"Skipping already processed tile: {dapi_path} and {collagen_path}")
            continue

        print(f"Processing: {dapi_path} and {collagen_path}")

        run_STIFMap(
            dapi=dapi_path,
            collagen=collagen_path,
            name='test',
            step=step,
            models=models,
            batch_size=batch_size,
            square_side=square_side,
            check_existing=True  # Set to True to check if the file already exists
        )

# Stitch the processed tiles back together
stitched_output_path = os.path.join(STIFMaps_directory, f"{base_name}_STIFMap_stitched.png")
stitch_images(stitched_output_path, base_name, num_rows, num_cols, image_format='png')
