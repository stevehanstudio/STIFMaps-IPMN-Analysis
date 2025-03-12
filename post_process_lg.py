from STIFMaps import STIFMap_generation
from STIFMaps.misc import get_step

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy import interpolate
from skimage import io
from skimage.transform import resize
from PIL import Image
import time
import psutil

# Increase the decompression limit
Image.MAX_IMAGE_PIXELS = None  # No limit

def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

project_dir = '/home/steve/Projects/WeaverLab/STIFMaps'
IPMN_directory = os.path.join(project_dir, "IPMN_images")
STIFMaps_directory = os.path.join(project_dir, "STIFMap_images")
os.makedirs(STIFMaps_directory, exist_ok=True)

# Path to original DAPI and Collagen stained images
orig_dapi = '/home/steve/Projects/WeaverLab/STIFMaps/IPMN_images/27620_C0_full.tif'
orig_collagen = '/home/steve/Projects/WeaverLab/STIFMaps/IPMN_images/27620_C1_full.tif'

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
scale_factor = 0.5

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
        # Open the image using Pillow
        with Image.open(image_path) as img:
            width, height = img.size
            print(f"File: {os.path.basename(image_path)}, Dimensions: {width}x{height}")
            return width, height
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None, None

def resize_crop_image(input_filename, base_name, orig_dapi):
    orig_width, orig_height = check_image_dimensions(orig_dapi)
    if orig_width is None or orig_height is None:
        print("Unable to get dimension of original DAPI file")
        return

    print(f"Original DAPI file dimension is {orig_width} by {orig_height}")

    # Define the total width and height for the stitched image
    total_width = 5003 * 7
    total_height = 5003 * 6

    # Open the stitched image
    stitched_image = Image.open(input_filename)

    print_memory_usage()  # Debug memory usage

    # Resize the stitched image to the total width and height
    resized_image = stitched_image.resize((total_width, total_height), Image.LANCZOS)

    print_memory_usage()  # Debug memory usage

    # Save a copy of the resized image
    resized_image_filename = os.path.join('/home/steve/Projects/WeaverLab/STIFMaps', f"{base_name}_resized.png")
    resized_image.save(resized_image_filename, format='PNG')
    resized_width, resized_height = check_image_dimensions(resized_image_filename)
    print(f"Resized image saved as {resized_image_filename} at {resized_width}x{resized_height}")

    # Crop the resized image to match the dimensions of the original DAPI image
    left = (total_width - orig_width) // 2
    upper = (total_height - orig_height) // 2
    right = left + orig_width
    lower = upper + orig_height
    cropped_image = resized_image.crop((left, upper, right, lower))

    # Save the cropped image
    output_filename = os.path.join('/home/steve/Projects/WeaverLab/STIFMaps', f"{base_name}_cropped.png")
    cropped_image.save(output_filename, format='PNG')
    print(f"Cropped image saved as {output_filename}")

# Example usage
stitched_image = '/home/steve/Projects/WeaverLab/STIFMaps/27620_STIFMap_stitched.png'
resize_crop_image(stitched_image, base_name, orig_dapi)
