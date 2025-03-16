import os
import re
import numpy as np
import tifffile
import matplotlib.pyplot as plt

# Constants - Directories, Global Variables
PROJECT_DIR = os.getcwd()
ORIG_IMAGE_DIR = os.path.join(PROJECT_DIR, 'IPMN_images')
MODELS_DIR = os.path.join(PROJECT_DIR, '../STIFMap_dataset/trained_models')

TEMP_OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'temp_outputs')
FINAL_OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'final_outputs')

BASE_NAMES = ['4601', '13401', '15806', '5114', '1865', '27620']
TILE_SIZE = 5003

# Helper Functions
def normalize_image(image, lower_percentile=1, upper_percentile=99):
    """Normalize the image to the range [0, 255] using percentile-based min and max values."""
    min_val = np.percentile(image, lower_percentile)
    max_val = np.percentile(image, upper_percentile)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

def plot_histogram(image, title):
    """Plot histogram of pixel values."""
    plt.hist(image.ravel(), bins=256, range=(image.min(), image.max()))
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.show()

def get_dapi_and_collagen_paths(base_name, orig_image_dir):
    """Find and return paths to DAPI and Collagen files."""
    dapi_pattern = re.compile(rf"^{base_name}_C0.*\.(tif|tiff)$", re.IGNORECASE)
    collagen_pattern = re.compile(rf"^{base_name}_C1.*\.(tif|tiff)$", re.IGNORECASE)
    dapi_path, collagen_path = None, None
    for file_name in os.listdir(orig_image_dir):
        if dapi_pattern.match(file_name):
            dapi_path = os.path.join(orig_image_dir, file_name)
        elif collagen_pattern.match(file_name):
            collagen_path = os.path.join(orig_image_dir, file_name)
    return dapi_path, collagen_path

def get_tile_base_name(input_path):
    # Extract the file name without the extension
    file_name = os.path.splitext(os.path.basename(input_path))[0]
    # Split by underscores and take the first two parts to get '27620_C0'
    base_name = "_".join(file_name.split("_")[:2])
    return base_name

# Function to check image dimensions
def check_image_dimensions(image_path):
    try:
        with tifffile.TiffFile(image_path) as tif:
            width, height = tif.pages[0].shape[:2]
            print(f"File: {os.path.basename(image_path)}, Dimensions: {width}x{height}")
            return width, height
    except Exception as e:
        print(f"Error opening {image_path}: {e}")

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

def get_base_name(file_path):
    """
    Extracts the base name (first part before '_') from a file name or full file path.
    
    Parameters:
        file_path (str): The file name or full file path.

    Returns:
        str: The base name (e.g., '27620').
    """
    # Extract just the file name from the full path
    file_name = os.path.basename(file_path)
    # Split the file name by underscores and return the first part
    base_name = file_name.split("_")[0]
    return base_name

# Function to generate the output path for STIFMap images
# def gen_output_path(filename):
#     pattern = r"_tile_(\d+)_(\d+)\.tif$"
#     match = re.search(pattern, filename)
#     if match:
#         row = int(match.group(1))
#         col = int(match.group(2))
#     else:
#         print("Error: row or column name not found in tiled C0 or C1 file name.")
#         return None

#     output_file = f"{base_name}_STIFMap_{row}_{col}.png"
#     return os.path.join(STIFMaps_directory, output_file)

def gen_STIFMap_tile_path(filename):
    """
    Generate the output path for STIFMap images.
    
    Parameters:
        filename (str): The name of the input tile file.
        base_name (str): The base name for the output file.
        STIFMaps_directory (str): The directory where the output file will be saved.

    Returns:
        str: The full output path for the STIFMap image, or None if row/column cannot be extracted.
    """
    # Pattern to match the row and column in the file name
    base_name = get_base_name(filename)
    pattern = r"_(\d+)_(\d+)\.tif$"
    match = re.search(pattern, filename)

    if not match:
        print(f"Error: Row or column not found in file name: {filename}")
        return None

    # Extract row and column numbers
    row = int(match.group(1))
    col = int(match.group(2))

    # Construct the output file name and path
    output_file = f"{base_name}_{row}_{col}.png"
    return os.path.join(TEMP_OUTPUTS_DIR, base_name, "STIFMap_tiles", output_file)