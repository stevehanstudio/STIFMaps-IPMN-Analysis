
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

project_dir = '/home/steve/Projects/WeaverLab/STIFMaps'
IPMN_directory = os.path.join(project_dir, "IPMN_images")
STIFMaps_directory = os.path.join(project_dir, "STIFMap_images_new")
os.makedirs(STIFMaps_directory, exist_ok=True)

# Define the base name and the number of rows and columns
base_name = "27620"
base_name_C0 = "27620_C0_full_tile"
base_name_C1 = "27620_C1_full_tile"
num_rows = 6  # Example: Number of rows in the grid of tiles
num_cols = 7  # Example: Number of columns in the grid of tiles

# Create a list of file names for C0 tiles
C0_files = [f"{base_name_C0}_{i}_{j}.tif" for i in range(num_rows) for j in range(num_cols)]

# Create a list of file names for C1 tiles
C1_files = [f"{base_name_C1}_{i}_{j}.tif" for i in range(num_rows) for j in range(num_cols)]


models = [
   '/home/steve/Projects/WeaverLab/STIFMap_dataset/trained_models/iteration_1171.pt',
   '/home/steve/Projects/WeaverLab/STIFMap_dataset/trained_models/iteration_1000.pt',
   '/home/steve/Projects/WeaverLab/STIFMap_dataset/trained_models/iteration_1043.pt',
   '/home/steve/Projects/WeaverLab/STIFMap_dataset/trained_models/iteration_1161.pt',
   '/home/steve/Projects/WeaverLab/STIFMap_dataset/trained_models/iteration_1180.pt']

# Networks were trained at a microscopy resolution of 4.160 pixels/micron (0.2404 microns/pixel)
# Provide a scale factor to resize the input images to this resolution
# Ex: Images at 2.308 pixels/micron require a scale_factor of 1.802
# scale_factor = 1.802
scale_factor = 2.712
# scale_factor = 0.5

# Stifness is predicted for each square. This is the distance from the center of one square to the next
step = 40

# How many squares to evaluate at once with the network
batch_size = 100

# Given the scale_factor, what is the actual step size (in pixels) from one square to the next?
step = get_step(step, scale_factor)

print('Step size is ' + str(step) + ' pixels')

# Get the actual side length of one square
# The models expect input squares that are 224 x 224 pixels. 
# Given the scale_factor, how many pixels is that in these images?
square_side = get_step(224, scale_factor)

print('Side length for a square is ' + str(square_side) + ' pixels')

def check_image_dimensions(image_path):
    try:
        # Open the image using tifffile
        with tifffile.TiffFile(image_path) as tif:
            width, height = tif.pages[0].shape[:2]
            print(f"File: {os.path.basename(image_path)}, Dimensions: {width}x{height}")
            return width, height
    except Exception as e:
        print(f"Error opening {image_path}: {e}")

def stitch_images(output_filename, base_name, rows, cols, image_format='png'):
    # Create a blank image to hold the stitched image
    # Assuming all images are of the same size
    sample_image_path = f"{base_name}_{0}_{0}.{image_format}"
    sample_image = Image.open(sample_image_path)
    image_width, image_height = sample_image.size

    # Calculate the size of the stitched image
    stitched_width = cols * image_width
    stitched_height = rows * image_height

    # Create a new blank image with the calculated size
    stitched_image = Image.new('RGB', (stitched_width, stitched_height))

    # Loop through the grid and paste each image into the stitched image
    for row in range(rows):
        for col in range(cols):
            # Construct the filename for the current image
            image_filename = f"{base_name}_{row}_{col}.{image_format}"
            # Open the image
            image = Image.open(image_filename)
            # Calculate the position to paste the image
            x = col * image_width
            y = row * image_height
            # Paste the image into the stitched image
            stitched_image.paste(image, (x, y))

    # Save the stitched image
    stitched_image.save(output_filename)
    print(f"Stitched image saved as {output_filename}")

def convert_seconds_to_hms(seconds):
    # Calculate hours, minutes, and seconds
    hours = int(seconds // 3600)  # Number of hours
    minutes = int((seconds % 3600) // 60)  # Number of minutes
    seconds = seconds % 60  # Remaining seconds (as a float)

    # Build the formatted string
    time_parts = []
    if hours > 0:
        time_parts.append(f"{hours} hours")
    if minutes > 0:
        time_parts.append(f"{minutes} minutes")
    time_parts.append(f"{seconds:.1f} seconds")

    # Join the parts into a single string
    return ", ".join(time_parts)

# Returns the path to the STIFMap image to be saved based on the file name of tiled image 
def gen_output_path(filename):
    # Define the regular expression pattern
    pattern = r"_tile_(\d+)_(\d+)\.tif$"
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    # Check if a match is found
    if match:
      # Extract the row and column indices from the match groups
      row = int(match.group(1))
      col = int(match.group(2))
    else:
      print("Error: row or column name not found in tiled C0 or C1 file name.")
    
    output_file = f"{base_name}_STIFMap_{row}_{col}.png"
    return os.path.join(STIFMaps_directory, output_file) 

# Generate and save the STIFMap given a given DAPI and Collagen image 
def run_STIFMap(dapi, collagen, name, step, models, batch_size, square_side):
   start_time = time.perf_counter()

   # Generate the stiffness predictions
   z_out = STIFMap_generation.generate_STIFMap(dapi, collagen, name, step, models=models,
                  mask=False, batch_size=batch_size, square_side=square_side,
                  save_dir=False)
    
   # Measure the elapsed time
   end_time = time.perf_counter()
   print("Elapsed time:", convert_seconds_to_hms(end_time - start_time))

   output_image = np.mean(z_out, axis=0)
   
   # Normalize the image to the range [0, 1]
  #  output_image_normalized = output_image / output_image.max()
   
   # Save the image using matplotlib
   output_path = gen_output_path(dapi)
   # '/home/steve/Projects/WeaverLab/STIFMaps/z_out/stiffness_map.png'
   #  plt.imsave(output_path, output_image_normalized, cmap="viridis")
   plt.imsave(output_path, output_image, cmap="viridis")
   print(f"Saved image: {output_path}")
   
   # Show the output image
   # `imshow` is deprecated since version 0.25 and will be removed in version 0.27. Using `matplotlib` in the next cell to visualize images.
   # io.imshow(np.mean(z_out, axis=0))

# Function to check if a tile has already been processed
def is_tile_completed(output_path):
    return os.path.exists(output_path)

# Main: Loop through all the tiled C0 and C1 images passes each one to run_STIFMap() 
for i in range(num_rows):
  for j in range(num_cols):
    dapi = os.path.join(IPMN_directory, 'tiled', C0_files[i * num_cols + j])
    collagen = os.path.join(IPMN_directory, 'tiled', C1_files[i * num_cols + j])
    output_path = gen_output_path(dapi)
    
    # Check if the tile has already been processed
    if is_tile_completed(output_path):
        print(f"Skipping already processed tile: {C0_files[i * num_cols + j]} and {C1_files[i * num_cols + j]}")
        continue
    
    print(f"Processing: {C0_files[i * num_cols + j]} and {C1_files[i * num_cols + j]}")

    run_STIFMap(
        dapi=dapi, 
        collagen=collagen, 
        name='test', 
        step=step, 
        models=models, 
        batch_size=batch_size, 
        square_side=square_side
    )
