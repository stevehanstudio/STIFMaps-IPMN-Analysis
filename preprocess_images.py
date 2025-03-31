import os
import numpy as np
import tifffile

# Import from the globals_and_helpers file
from globals_and_helpers import (
    PROJECT_DIR,
    ORIG_IMAGE_DIR,
    TEMP_OUTPUTS_DIR,
    TILE_SIZE,
    BASE_NAMES,
    normalize_image,
    plot_histogram,
    get_dapi_and_collagen_paths,
    get_tile_base_name,
    check_image_dimensions,
)

import os
import numpy as np
import tifffile

def gen_tile_images(input_path, output_dir, tile_size=5003):
    """
    Generate tiled images from a large TIFF image.

    Parameters:
    - input_path: Path to the input TIFF image.
    - output_dir: Directory to save the output tiles.
    - tile_size: Size of each tile.
    """
    # Check if the input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    try:
        # Open the large image using tifffile
        tif = tifffile.TiffFile(input_path)
        image = tif.asarray()
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    tile_image_dir = os.path.join(output_dir, "IPMN_tiles")
    os.makedirs(tile_image_dir, exist_ok=True)

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the number of tiles
    num_tiles_x = (width + tile_size - 1) // tile_size
    num_tiles_y = (height + tile_size - 1) // tile_size

    # Iterate over the image and crop tiles
    for i in range(num_tiles_y):
        for j in range(num_tiles_x):
            # Calculate the coordinates for the current tile
            left = j * tile_size
            upper = i * tile_size
            right = min(left + tile_size, width)
            lower = min(upper + tile_size, height)

            # Define the tile filename
            base_name = get_tile_base_name(input_path)
            tile_filename = os.path.join(tile_image_dir, f"{base_name}_{i}_{j}.tif")

            # Check if the tile already exists and is not empty
            if os.path.exists(tile_filename):
                if os.path.getsize(tile_filename) > 0:
                    print(f"Tile {tile_filename} already exists and is not empty. Skipping.")
                    continue
                else:
                    print(f"Tile {tile_filename} exists but is empty. Overwriting.")

            # Crop the tile
            tile = image[upper:lower, left:right]

            # Create a new image with the desired size
            new_tile = np.zeros((tile_size, tile_size), dtype=np.uint8)
            new_tile[:tile.shape[0], :tile.shape[1]] = tile

            # Save the tile using tifffile
            tifffile.imwrite(tile_filename, new_tile)
            print(f"Tile {tile_filename} created and saved.")

# Loop through all base names and generate tiles for corresponding images
for base_name in BASE_NAMES:
    # Use the function to get paths for DAPI and Collagen images
    dapi_path, collagen_path = get_dapi_and_collagen_paths(base_name, ORIG_IMAGE_DIR)
    
    # Verify that the files exist before proceeding
    if dapi_path is None:
        print(f"Warning: DAPI file for {base_name} not found.")
        continue
    if collagen_path is None:
        print(f"Warning: Collagen file for {base_name} not found.")
        continue

    # Create output directory for tiles for the current base name
    output_dir = os.path.join(TEMP_OUTPUTS_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Generate tiles for DAPI and Collagen images
    print(f"Processing DAPI image for base name {base_name}...")
    gen_tile_images(dapi_path, output_dir, tile_size=TILE_SIZE)
    print(f"Processing Collagen image for base name {base_name}...")
    gen_tile_images(collagen_path, output_dir, tile_size=TILE_SIZE)

    