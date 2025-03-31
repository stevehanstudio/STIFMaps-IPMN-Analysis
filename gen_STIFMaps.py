# Import from the globals_and_helpers file
from globals_and_helpers import (
    PROJECT_DIR,
    MODELS_DIR,
    ORIG_IMAGE_DIR,
    TEMP_OUTPUTS_DIR,
    FINAL_OUTPUTS_DIR,
    TILE_SIZE,
    BASE_NAMES,
    normalize_image,
    plot_histogram,
    get_dapi_and_collagen_paths,
    convert_seconds_to_hms,
    get_base_name,
    check_image_dimensions,
    gen_STIFMap_tile_path,
    # save_stiffness_colormap,
    stitch_STIFMap_tiles,
    # stitch_images,
    gen_colormap_legend,
)

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

# STIFMap models
models = [
    os.path.join(MODELS_DIR, 'iteration_1171.pt'),
    os.path.join(MODELS_DIR, 'iteration_1000.pt'),
    os.path.join(MODELS_DIR, 'iteration_1043.pt'),
    os.path.join(MODELS_DIR, 'iteration_1161.pt'),
    os.path.join(MODELS_DIR, 'iteration_1180.pt')
]

# Parameters
# STIFMap_SCALE_FACTOR = 0.4
STIFMap_SCALE_FACTOR = 2.712
STIFMap_STEP = get_step(40, STIFMap_SCALE_FACTOR)
STIFMap_SQUARE_SIDE = get_step(224, STIFMap_SCALE_FACTOR)
STIFMap_BATCH_SIZE = 100

print('Step size is ' + str(STIFMap_STEP) + ' pixels')
print('Side length for a square is ' + str(STIFMap_SQUARE_SIDE) + ' pixels')

def gen_STIFMap_tile(dapi_path, collagen_path, name, step, models, batch_size, square_side, check_existing=True):
    """
    Generate and save the STIFMap for a given tile.

    Parameters:
    - dapi_path: Path to the DAPI image.
    - collagen_path: Path to the collagen image.
    - name: Name associated with the STIFMap.
    - step: Step size for processing.
    - models: Models used for STIFMap generation.
    - batch_size: Batch size for processing.
    - square_side: Side length of the square for processing.
    - check_existing: Flag to check if the tile has already been processed.
    """
    output_path = gen_STIFMap_tile_path(dapi_path)

    # Check if the tile has already been processed
    if check_existing and os.path.exists(output_path):
        print(f"Skipping already processed tile: {dapi_path} and {collagen_path}")
        return

    start_time = time.perf_counter()

    z_out = STIFMap_generation.generate_STIFMap(
        dapi=dapi_path,
        collagen=collagen_path,
        name=name,
        step=step,
        models=models,
        mask=False,
        batch_size=batch_size,
        square_side=square_side,
        save_dir=False
    )

    end_time = time.perf_counter()
    print("Elapsed time:", convert_seconds_to_hms(end_time - start_time))

    # Calculate the mean output image along the first axis
    output_image = np.mean(z_out, axis=0)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the output image without normalization
    plt.imsave(output_path, output_image, cmap="viridis")
    print(f"Saved image: {output_path}")

    # Save the raw stiffness values as a NumPy array
    stiffness_values_path = output_path.replace(".png", ".npy")
    np.save(stiffness_values_path, output_image)
    print(f"Saved stiffness values: {stiffness_values_path}")

# Function to generate and save the STIFMap
# def gen_STIFMap_tile(dapi_path, collagen_path, name, step, models, batch_size, square_side, check_existing=True):
#     output_path = gen_STIFMap_tile_path(dapi_path)
    
#     # Check if the tile has already been processed
#     if check_existing and os.path.exists(output_path):
#         print(f"Skipping already processed tile: {dapi_path} and {collagen_path}")
#         return

#     start_time = time.perf_counter()

#     z_out = STIFMap_generation.generate_STIFMap(
#         dapi=dapi_path, 
#         collagen=collagen_path, 
#         name=name, 
#         step=step, 
#         models=models,
#         mask=False, 
#         batch_size=batch_size, 
#         square_side=square_side,
#         save_dir=False
#     )

#     end_time = time.perf_counter()
#     print("Elapsed time:", convert_seconds_to_hms(end_time - start_time))

#     output_image = np.mean(z_out, axis=0)

#     global_min = np.min(output_image)
#     global_max = np.max(output_image)
#     output_image_normalized = (output_image - global_min) / (global_max - global_min)

#     # Ensure the output directory exists
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)

#     plt.imsave(output_path, output_image_normalized, cmap="viridis")
#     print(f"Saved image: {output_path}")

#     # Save the raw stiffness values as a NumPy array
#     stiffness_values_path = output_path.replace(".png", ".npy")
#     np.save(stiffness_values_path, output_image)
#     print(f"Saved stiffness values: {stiffness_values_path}")

# Function to check if a tile has already been processed
def is_tile_completed(output_path):
    return os.path.exists(output_path)

def get_base_file_name(file_path):
    # Extract the file name without the extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    # Keep only the first two parts (e.g., '27620_C0') by splitting on '_'
    base_name = "_".join(file_name.split("_")[:2])
    return base_name

# Generate the STIFMaps for all the tiles for a base image
def gen_STIFMap(base_name):
    dapi_path, collagen_path = get_dapi_and_collagen_paths(base_name, orig_image_dir=ORIG_IMAGE_DIR)
    base_name_C0 = get_base_file_name(dapi_path)
    base_name_C1 = get_base_file_name(collagen_path)    
    
    # Automatically determine num_rows and num_cols based on filenames
    tile_pattern_C0 = re.compile(rf"{base_name_C0}_(\d+)_(\d+)\.tif")
    tile_pattern_C1 = re.compile(rf"{base_name_C1}_(\d+)_(\d+)\.tif")

    # tile_pattern = re.compile(rf"{base_name_C0}_(\d+)_(\d+)\.tif")
    tile_image_dir = os.path.join(TEMP_OUTPUTS_DIR, base_name, "IPMN_tiles")
    
    # Dictionary to store how many columns exist per row
    row_col_map = {}
    
    # Loop through files and match patterns
    for file in os.listdir(tile_image_dir):
        # match = tile_pattern.match(file)
        match_C0 = tile_pattern_C0.match(file)
        match_C1 = tile_pattern_C1.match(file)

        if match_C0 or match_C1:
            match = match_C0 or match_C1
            row, col = map(int, match.groups())
            row_col_map.setdefault(row, set()).add(col)
            # print(f"Matched Row: {row}, Column: {col}")
    
    # Find the maximum number of columns
    if not row_col_map:
        raise ValueError("No matching files found in the directory.")
    
    max_cols = max(len(cols) for cols in row_col_map.values())
    
    # Select only rows with the full set of columns
    valid_rows = sorted([row for row, cols in row_col_map.items() if len(cols) == max_cols])
    num_rows = len(valid_rows)
    num_cols = max_cols

    print(f"Detected grid size: {num_rows} rows x {num_cols} columns")
    
    # Main: Loop through all the tiled C0 and C1 images and pass each one to run_STIFMap()
    for row in valid_rows:  # Iterate only over rows with full columns
        for col in range(num_cols):
            dapi_path = os.path.join(tile_image_dir, f"{base_name_C0}_{row}_{col}.tif")
            collagen_path = os.path.join(tile_image_dir, f"{base_name_C1}_{row}_{col}.tif")
    
            # Ensure files exist before processing
            if not os.path.exists(dapi_path) or not os.path.exists(collagen_path):
                print(f"Skipping missing tile: {dapi_path} or {collagen_path}")
                continue
            
            output_path = gen_STIFMap_tile_path(dapi_path)
            # output_path = gen_output_path(dapi_path)
            
            if is_tile_completed(output_path):
                print(f"Skipping already processed tile: {dapi_path} and {collagen_path}")
                continue

            print(f"Generating STIFMap for {dapi_path}, {collagen_path}")
            gen_STIFMap_tile(
                dapi_path, collagen_path, name=base_name, step=STIFMap_STEP,
                models=models, batch_size=STIFMap_BATCH_SIZE, square_side=STIFMap_SQUARE_SIDE,
                check_existing=True
            )
    
    stitched_output_path = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_STIFMap_stitched.png")
    stitch_STIFMap_tiles(base_name)
    # save_stiffness_colormap(stitched_output_path, base_name)
    gen_colormap_legend(base_name)

for base_name in BASE_NAMES:
    gen_STIFMap(base_name=base_name)