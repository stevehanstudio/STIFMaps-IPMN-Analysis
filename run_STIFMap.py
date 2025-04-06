# Import from the globals_and_helpers file
from globals_and_helpers import (
    PROJECT_DIR,
    MODELS_DIR,
    ORIG_IMAGE_DIR,
    TEMP_OUTPUTS_DIR,
    FINAL_OUTPUTS_DIR,
    TILE_SIZE,
    BASE_NAMES,
    get_dapi_and_collagen_paths,
    convert_seconds_to_hms,
    get_base_name,
    check_image_dimensions,
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
# STIFMap_SCALE_FACTOR = 0.5
STIFMap_SCALE_FACTOR = 2.712
STIFMap_STEP = get_step(40, STIFMap_SCALE_FACTOR)
STIFMap_SQUARE_SIDE = get_step(224, STIFMap_SCALE_FACTOR)
STIFMap_BATCH_SIZE = 100

print('Step size is ' + str(STIFMap_STEP) + ' pixels')
print('Side length for a square is ' + str(STIFMap_SQUARE_SIDE) + ' pixels')

# for i in range(len(dap_files)):
def run_STIFMAP(base_name):
    start_time = time.perf_counter()
    dapi_path, collagen_path = get_dapi_and_collagen_paths(base_name, ORIG_IMAGE_DIR)
    base_name = get_base_name(dapi_path)
    output_path = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_STIFMap.png")

   # Check if the tile has already been processed
    if not os.path.exists(dapi_path) or not os.path.exists(collagen_path):
        print(f"File not found: {dapi_path} and {collagen_path}")
        exit
    print(f"Processing: {dapi_path}, {collagen_path}")

    try:
        z_out = STIFMap_generation.generate_STIFMap(
            dapi=dapi_path,
            collagen=collagen_path,
            name="test",
            step=STIFMap_STEP,
            models=models,
            mask=False,
            batch_size=STIFMap_BATCH_SIZE,
            square_side=STIFMap_SQUARE_SIDE,
            save_dir=False
        )
    except Exception as e:  # Catch all exceptions
        print(f"Error occurred: {e}.  Continuing with next image")
        return  # Skip the current iteration and move to the next item

    end_time = time.perf_counter()
    print("Elapsed time:", convert_seconds_to_hms(end_time - start_time))
    output_image = np.mean(z_out, axis=0)

    # Calculate raw min and max stiffness values
    min_stiffness = np.min(output_image)
    max_stiffness = np.max(output_image)

    # Calculate the conversion factor
    conversion_factor = (max_stiffness - min_stiffness) / 255

    # Print the statistics
    print(f"Image: {output_path}")
    print(f"Min Stiffness: {min_stiffness}")
    print(f"Max Stiffness: {max_stiffness}")
    print(f"Conversion Factor: {conversion_factor}")
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the output image without normalization
    plt.imsave(output_path, output_image, cmap="gray")
    print(f"Saved image: {output_path}")

for base_name in BASE_NAMES:
    run_STIFMAP(base_name)