import os
import re
import numpy as np
import pandas as pd
import tifffile
import json
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from shapely.geometry import shape, mapping, Polygon
from shapely.affinity import scale

# Constants - Directories, Global Variables
NO_TILING = True
PROJECT_DIR = os.getcwd()
FINAL_OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'final_outputs')
TEMP_OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'temp_outputs')
MODELS_DIR = os.path.join(PROJECT_DIR, '../STIFMap_dataset/trained_models')
QUPATH_PROJECT_DIR = os.path.join(PROJECT_DIR, '../analysis_panel_1')
if NO_TILING:
    ORIG_IMAGE_DIR = os.path.join(TEMP_OUTPUTS_DIR, 'resized0.25_IPMN_images')
    TILE_SIZE = None
else:
    ORIG_IMAGE_DIR = os.path.join(PROJECT_DIR, 'IPMN_images')
    TILE_SIZE = 5003

BASE_NAMES = ['8761', '9074',]
# BASE_NAMES = ['13401']
# BASE_NAMES = ['1865', '5114', '5789', '6488', '8761', '9074', '13401', '15806']
# BASE_NAMES = ['7002', '27620', '15806', '4601', '13401', '5114', '1865']

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

def check_image_dimensions(image_path):
    try:
        if image_path.lower().endswith(('.tif', '.tiff')):
            # Open the TIFF image using tifffile
            with tifffile.TiffFile(image_path) as tif:
                height, width = tif.pages[0].shape[:2]
        elif image_path.lower().endswith('.png'):
            # Open the PNG image using PIL
            with Image.open(image_path) as img:
                width, height = img.size
        else:
            raise ValueError("Unsupported image format. Only TIFF and PNG are supported.")

        print(f"File: {os.path.basename(image_path)}, Dimensions: {width}x{height}")
        return width, height
    except Exception as e:
        print(f"Error opening {image_path}: {e}")
        return None

# Don't use
def resize_and_square_image(image_path):
    # Open the image using tifffile
    with tifffile.TiffFile(image_path) as tif:
        image = tif.asarray()  # Read the image as-is

    # Ensure the image is interpreted as 16-bit unsigned
    image = image.astype(np.uint16)

    # Print the starting dimensions of the image
    original_height, original_width = image.shape[:2]
    print(f"Starting dimensions of the image: {original_width}x{original_height}")

    # Resize the image to 1/5 (20%) of the original size
    new_width = int(original_width * 0.2)
    new_height = int(original_height * 0.2)

    # Resize the image using PIL while preserving the 16-bit depth
    resized_image = Image.fromarray(image).resize((new_width, new_height), Image.LANCZOS)

    # Convert the resized image back to a NumPy array with 16-bit unsigned type
    resized_image_array = np.array(resized_image, dtype=np.uint16)

    # Find the darkest color in the grayscale image
    darkest_color = np.min(image)

    # Create a new square image with the darkest color
    square_size = max(new_width, new_height)
    square_image_array = np.full((square_size, square_size), darkest_color, dtype=np.uint16)

    # Paste the resized image onto the square image at the top-left corner
    square_image_array[:new_height, :new_width] = resized_image_array

    # Convert the square image array back to a PIL Image
    square_image = Image.fromarray(square_image_array)

    # Return the square image
    return square_image

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

def save_stiffness_colormap(stiffness_map, save_path):
    """
    Save a colorbar legend for the given stiffness map.

    Parameters:
    - stiffness_map (np.ndarray): The stiffness data array.
    - save_path (str): The path to save the colorbar legend image.
    """
    # Extract min and max from the actual data
    min_value = np.min(stiffness_map)
    max_value = np.max(stiffness_map)

    # Create a dummy image for the colormap
    fig, ax = plt.subplots(figsize=(2, 6))
    dummy_img = ax.imshow([[min_value, max_value]], cmap='viridis', aspect='auto')

    # Remove the dummy axes
    ax.remove()

    # Create colorbar with proper labeling
    cbar = fig.colorbar(dummy_img, ax=fig.add_subplot(111))
    cbar.set_label("Stiffness (Young's Modulus) [kPa]", fontsize=14)

    # Save and close
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def gen_colormap_legend(base_name):
    """
    Generate and save a colormap legend for the raw stiffness values.

    Parameters:
    - base_name: The base name to determine the directory.
    """
    stiffness_dir = os.path.join(TEMP_OUTPUTS_DIR, base_name, "STIFMap_tiles")

    # Initialize min and max values
    min_value = float('inf')
    max_value = float('-inf')

    # Initialize a counter for the number of .npy files found
    npy_file_count = 0

    # Find all .npy files in the directory
    for file_name in os.listdir(stiffness_dir):
        if file_name.endswith('.npy'):
            npy_path = os.path.join(stiffness_dir, file_name)
            try:
                stiffness_map = np.load(npy_path)

                # Update min and max values
                current_min = np.min(stiffness_map)
                current_max = np.max(stiffness_map)
                min_value = min(min_value, current_min)
                max_value = max(max_value, current_max)

                # Increment the counter
                npy_file_count += 1
            except FileNotFoundError:
                print(f"Warning: Could not find .npy file at {npy_path}")
            except Exception as e:
                print(f"Warning: Error loading .npy file at {npy_path}: {e}")

    # Print the number of .npy files found
    print(f"Found {npy_file_count} .npy files in {stiffness_dir}")

    # Print the smallest min_value and largest max_value found
    if npy_file_count > 0:
        print(f"Smallest min_value found: {min_value}")
        print(f"Largest max_value found: {max_value}")
    else:
        print("No .npy files found, so no min/max values to display.")
        return  # Exit the function if no .npy files are found

    # Create a dummy image for the colormap
    fig, ax = plt.subplots(figsize=(2, 6))
    dummy_img = ax.imshow([[min_value, max_value]], cmap='viridis', aspect='auto')

    # Remove the dummy axes
    ax.remove()

    # Create colorbar with proper labeling
    cbar = fig.colorbar(dummy_img, ax=fig.add_subplot(111))
    cbar.set_label("Stiffness (Young's Modulus) [kPa]", fontsize=14)

    # Generate intermediate tick values (whole numbers)
    ticks = [min_value]
    num_intermediate_ticks = 8
    if max_value > min_value:
        import math
        start_val = math.ceil(min_value)
        end_val = math.floor(max_value)
        intermediate_ticks = [val for val in range(start_val, end_val + 1)]

        # Select up to 8 intermediate ticks
        if len(intermediate_ticks) <= num_intermediate_ticks:
            ticks.extend(intermediate_ticks)
        else:
            step = max(1, len(intermediate_ticks) // num_intermediate_ticks)
            ticks.extend(intermediate_ticks[::step][:num_intermediate_ticks])

    ticks.append(max_value)
    unique_ticks = sorted(list(set(ticks))) # Ensure unique and sorted ticks

    # Format tick labels
    tick_labels = []
    for tick in unique_ticks:
        if tick == min_value or tick == max_value:
            tick_labels.append(f"{tick:.3g}") # 3 significant digits for bounds
        elif tick == int(tick): # Check if it's a whole number
            tick_labels.append(f"{tick:.2g}") # 2 significant digits for whole numbers
        else:
            tick_labels.append("") # Don't label non-whole intermediate ticks

    cbar.set_ticks(unique_ticks)
    cbar.set_ticklabels(tick_labels)

    # Save the colormap legend
    legend_path = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_stiffness_colormap_legend.png")
    plt.savefig(legend_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Colormap legend saved as {legend_path}")

def stitch_STIFMap_tiles(base_name, file_extension='npy'):
    os.makedirs(TEMP_OUTPUTS_DIR, exist_ok=True)
    os.makedirs(FINAL_OUTPUTS_DIR, exist_ok=True)

    # Dynamically determine the number of rows and columns
    tile_pattern = re.compile(rf"{base_name}_(\d+)_(\d+)\.{file_extension}")
    STIFMaps_directory = os.path.join(TEMP_OUTPUTS_DIR, base_name, "STIFMap_tiles")
    output_filename = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_STIFMap_stitched.png")

    row_col_map = {}
    for file in os.listdir(STIFMaps_directory):
        match = tile_pattern.match(file)
        if match:
            row, col = map(int, match.groups())
            row_col_map.setdefault(row, set()).add(col)

    if not row_col_map:
        raise ValueError("No matching files found in the directory.")

    # Correctly determine the number of rows and columns
    max_row = max(row_col_map.keys())
    max_col = max(max(cols) for cols in row_col_map.values())
    num_rows = max_row + 1
    num_cols = max_col + 1

    print(f"Detected grid size: {num_rows} rows x {num_cols} columns")

    # Determine tile dimensions from the first available .npy file
    tile_height = tile_width = None
    for row in range(num_rows):
        for col in range(num_cols):
            file_name = f"{base_name}_{row}_{col}.{file_extension}"
            file_path = os.path.join(STIFMaps_directory, file_name)
            if os.path.exists(file_path):
                tile = np.load(file_path)
                tile_height, tile_width = tile.shape
                break
        if tile_height is not None:
            break

    if tile_height is None:
        raise ValueError("No valid .npy files found to determine dimensions.")

    # Create an empty array for the stitched image
    stitched_height = num_rows * tile_height
    stitched_width = num_cols * tile_width
    stitched_image = np.zeros((stitched_height, stitched_width))

    # Stitch the tiles together
    for row in range(num_rows):
        for col in range(num_cols):
            file_name = f"{base_name}_{row}_{col}.{file_extension}"
            file_path = os.path.join(STIFMaps_directory, file_name)

            if os.path.exists(file_path):
                tile = np.load(file_path)
            else:
                tile = np.zeros((tile_height, tile_width))
                print(f"Missing tile: {file_name}. Replacing with a zero tile.")

            x = col * tile_width
            y = row * tile_height
            stitched_image[y:y + tile_height, x:x + tile_width] = tile

    # Calculate the conversion factor
    min_val = np.min(stitched_image)
    max_val = np.max(stitched_image)
    conversion_factor = 255 / (max_val - min_val)
    print(f"Conversion factor: {conversion_factor}")

    # Normalize the stitched image to the range [0, 255]
    normalized_image = ((stitched_image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    # Save the grayscale image
    plt.imsave(output_filename, normalized_image, cmap='gray')
    print(f"Stitched grayscale image saved as {output_filename}")


# def stitch_STIFMap_tiles(base_name, image_format='png'):
#     os.makedirs(TEMP_OUTPUTS_DIR, exist_ok=True)
#     os.makedirs(FINAL_OUTPUTS_DIR, exist_ok=True)

#     # Dynamically determine the number of rows and columns
#     tile_pattern = re.compile(rf"{base_name}_(\d+)_(\d+)\.{image_format}")
#     STIFMaps_directory = os.path.join(TEMP_OUTPUTS_DIR, base_name, "STIFMap_tiles")
#     output_filename = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_STIFMap_stitched.png")
    
#     row_col_map = {}
#     for file in os.listdir(STIFMaps_directory):
#         match = tile_pattern.match(file)
#         if match:
#             row, col = map(int, match.groups())
#             row_col_map.setdefault(row, set()).add(col)

#     if not row_col_map:
#         raise ValueError("No matching files found in the directory.")

#     # Correctly determine the number of rows and columns
#     max_row = max(row_col_map.keys())
#     max_col = max(max(cols) for cols in row_col_map.values())
#     num_rows = max_row + 1
#     num_cols = max_col + 1

#     print(f"Detected grid size: {num_rows} rows x {num_cols} columns")

#     # Determine image dimensions from the first available .png tile
#     image_width = image_height = None
#     for row in range(num_rows):
#         for col in range(num_cols):
#             image_filename = f"{base_name}_{row}_{col}.{image_format}"
#             image_path = os.path.join(STIFMaps_directory, image_filename)
#             if os.path.exists(image_path):
#                 try:
#                     with Image.open(image_path) as image:
#                         image_width, image_height = image.size
#                     break
#                 except Exception as e:
#                     print(f"Error opening {image_path}: {e}")
#         if image_width is not None:
#             break

#     if image_width is None:
#         raise ValueError("No valid .png image files found to determine dimensions.")

#     stitched_width = num_cols * image_width
#     stitched_height = num_rows * image_height
#     stitched_image = Image.new('RGB', (stitched_width, stitched_height), color='white')

#     for row in range(num_rows):
#         for col in range(num_cols):
#             image_filename = f"{base_name}_{row}_{col}.{image_format}"
#             image_path = os.path.join(STIFMaps_directory, image_filename)

#             if os.path.exists(image_path):
#                 try:
#                     image = Image.open(image_path)
#                 except Exception as e:
#                     print(f"Error opening {image_path}: {e}")
#                     image = Image.new('RGB', (image_width, image_height), color='white')
#                     print(f"Missing tile: {image_filename}. Replacing with a white tile.")
#             else:
#                 image = Image.new('RGB', (image_width, image_height), color='white')
#                 print(f"Missing tile: {image_filename}. Replacing with a white tile.")

#             x = col * image_width
#             y = row * image_height
#             stitched_image.paste(image, (x, y))

#     stitched_image.save(output_filename)
#     print(f"Stitched image saved as {output_filename}")

def calc_crop_dimensions(base_name, image_format='png'):
    """
    Calculate the dimensions for cropping the STIFMap image based on a scaling factor.

    Parameters:
    - base_name: The base name used to construct file paths.
    - image_format: The format of the image file (default is 'png').

    Returns:
    - cropped_width: The calculated width for cropping, rounded to the nearest integer.
    - cropped_height: The calculated height for cropping, rounded to the nearest integer.
    """
    STIFMap_tile_image_path = os.path.join(TEMP_OUTPUTS_DIR, base_name, 'STIFMap_tiles', f'{base_name}_0_0.png')
    # STIFMap_tile_image_path = os.path.join(TEMP_OUTPUTS_DIR, base_name, 'STIFMap_tiles', f'{base_name}_STIFMap_0_0.png')

    # Calculate the scaling factor
    try:
        with Image.open(STIFMap_tile_image_path) as tile_image:
            scaling_factor = TILE_SIZE / tile_image.size[0]
    except FileNotFoundError:
        print(f"Error: Tile image not found at {STIFMap_tile_image_path}")
        return None, None
    
    dapi_path, collagen_path = get_dapi_and_collagen_paths(base_name, ORIG_IMAGE_DIR)
    orig_width, orig_height = check_image_dimensions(dapi_path)

    # Define the path to the STIFMap image file
    STIFMap_image_path = os.path.join(FINAL_OUTPUTS_DIR, f'{base_name}_STIFMap_stitched.png')

    # Load the STIFMap image
    with Image.open(STIFMap_image_path) as STIFMap_image:
        # Get the dimensions of the STIFMap image
        width, height = STIFMap_image.size

    # Calculate the dimensions for cropping
    cropped_width = round(orig_width / scaling_factor)
    cropped_height = round(orig_height / scaling_factor)

    print(f"Original DAPI Image Dimensions: {orig_width}x{orig_height}")
    print(f"Original STIFMap Image Dimensions: {width}x{height}")
    print(f"Cropped Image Dimensions: {cropped_width}x{cropped_height}")

    print(f"orig_width: {orig_width}, orig_height: {orig_height}, scaling_factor: {scaling_factor}")

    return cropped_width, cropped_height

def crop_STIFMap(base_name, image_format='png'):

    # Define the path to the image file
    image_path = os.path.join(FINAL_OUTPUTS_DIR, f'{base_name}_STIFMap_stitched.png')

    # Load the original image
    original_image = Image.open(image_path)

    # Define the dimensions for cropping
    # cropped_width = 2209
    # cropped_height = 1824
    cropped_width, cropped_height = calc_crop_dimensions(base_name)

    # Crop the image from the top-left corner
    cropped_image = original_image.crop((0, 0, cropped_width, cropped_height))

    # Convert the cropped image to grayscale
    gray_image = cropped_image.convert("L")

    # Define the path and name for the saved image
    save_path = os.path.join(FINAL_OUTPUTS_DIR, f'{base_name}_STIFMap_stitched_cropped_gray.png')
 
    # Save the grayscale image as PNG
    gray_image.save(save_path)

    print(f"Grayscale image saved to {save_path}")

def scale_annotations(base_name, resized=False):
    """
    Scales annotations in a GeoJSON file and calculates mean intensities from an image."
    """

    # File paths
    input_geojson_path = os.path.join(QUPATH_PROJECT_DIR, f"{base_name}_annotations.geojson")
    if resized:
        output_geojson_path = os.path.join(QUPATH_PROJECT_DIR, f"{base_name}_scaled_annotations_0.2.geojson")
    else:
        output_geojson_path = os.path.join(QUPATH_PROJECT_DIR, f"{base_name}_scaled_annotations.geojson")

    # Load the original GeoJSON file
    with open(input_geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Debugging: Print entire GeoJSON structure
    # print("Loaded GeoJSON Data:")
    # print(json.dumps(geojson_data, indent=2))

    # Debugging: Print a summary of the GeoJSON structure
    print(f"Number of features in GeoJSON: {len(geojson_data['features'])}")

    dapi_path, collagen_path = get_dapi_and_collagen_paths(base_name, ORIG_IMAGE_DIR)
    orig_width, orig_height = check_image_dimensions(dapi_path)
    if resized:
        STIFMap_width, STIFMap_height = check_image_dimensions(os.path.join(FINAL_OUTPUTS_DIR, f'{base_name}_STIFMap_resized.png'))
    else:
        STIFMap_width, STIFMap_height = calc_crop_dimensions(base_name)

    # Scaling factors for width and height
    # xfact = 2209 / 31398  # Approximately 0.0704
    # yfact = 1824 / 25922  # Approximately 0.0704
    xfact = STIFMap_width / orig_width  # Approximately 0.0704
    yfact = STIFMap_height / orig_height  # Approximately 0.0704

    # Check each feature and print only its keys
    for i, feature in enumerate(geojson_data['features']):
        # print(f"Feature {i} Keys:")
        # print(feature.keys())  # Print top-level keys of the feature
        # print("Properties Keys:")
        # print(feature.get('properties', {}).keys())  # Print keys within 'properties'

        # Check for missing 'classification' key
        if 'classification' not in feature.get('properties', {}):
            print(f"Feature {i} with ID {feature.get('id', 'Unknown')} is missing 'classification'.")

    # Scale each feature's geometry
    for feature in geojson_data['features']:
        # Debugging: Print feature properties
        # print("Processing Feature:")
        # print(json.dumps(feature, indent=2))
        
        # Check if 'classification' exists in properties
        if 'classification' not in feature['properties']:
            print(f"Feature with ID {feature.get('id', 'Unknown')} is missing 'classification'. Skipping.")
            feature['properties']['classification'] = {}  # Default to empty dictionary
        
        # Safely handle 'classification'
        classification = feature['properties']['classification']
        feature['properties']['Classification'] = classification.get('name', 'Unknown')
        
        # Update other properties
        feature['properties']['Object ID'] = feature.get('id', 'Unknown')
        feature['properties']['ROI'] = feature['geometry']['type']  # Should reflect Polygon type

        # Scale geometry
        geometry = shape(feature['geometry'])  # Convert GeoJSON geometry to Shapely object
        scaled_geometry = scale(geometry, xfact=xfact, yfact=yfact, origin=(0, 0))  # Scale the geometry
        feature['geometry'] = mapping(scaled_geometry)  # Update GeoJSON with scaled geometry

    # Save scaled annotations to GeoJSON
    with open(output_geojson_path, 'w') as f:
        json.dump(geojson_data, f, indent=2)

    print(f"Scaled annotations saved to: {output_geojson_path}")

def gen_report(base_name):
    """
    Generates a report by analyzing mean intensities from a GeoJSON file and an image.
    """

    # File paths
    input_geojson_path = os.path.join(QUPATH_PROJECT_DIR, f"{base_name}_annotations.geojson")
    image_path = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_STIFMap_stitched_cropped_gray.png")
    output_csv_path = os.path.join(QUPATH_PROJECT_DIR, f"{base_name}_mean_intensity_results.csv")

    # Load the original GeoJSON file
    with open(input_geojson_path, 'r') as f:
        geojson_data = json.load(f)

    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Analyze mean intensities
    results = []

    for feature in geojson_data['features']:
        object_id = feature['properties'].get('Object ID', 'Unknown')
        roi = feature['properties'].get('ROI', 'Unknown')
        classification = feature['properties'].get('Classification', 'Unknown')
        
        # Convert scaled geometry to mask
        polygon = shape(feature['geometry'])
        mask = np.zeros(image.shape, dtype=np.uint8)
        
        if isinstance(polygon, Polygon):
            points = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)
        else:
            print(f"Skipping non-polygon geometry for Object ID {object_id}")
            continue
        
        # Calculate mean intensity
        mean_intensity = cv2.mean(image, mask=mask)[0]
        
        # Store result
        results.append({
            "Object ID": object_id,
            "ROI": roi,
            "Classification": classification,
            "Mean Intensity": mean_intensity
        })

    # Export results to CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)

    print(f"Mean intensities saved to: {output_csv_path}")

    # File path for the summary table image
    output_image_path = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_summary_table.jpeg")

    # Read the generated CSV into a DataFrame
    summary_df = pd.read_csv(output_csv_path)

    # Summarize the data by classification
    summary = df.groupby('Classification').agg(
        Total_Number=('Classification', 'size'),
        Mean_Intensity=('Mean Intensity', 'mean')
    ).reset_index()

    # File path for the summary table image
    output_image_path = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_summary_table2.jpeg")

    # Create a summary table visualization
    fig, ax = plt.subplots(figsize=(8, 5))  # Adjust the size as needed
    ax.axis('off')  # Hide the axes

    # Add a table to the figure
    table = plt.table(cellText=summary.values, colLabels=summary.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(summary.columns))))

    # Save the table as a JPEG image
    plt.savefig(output_image_path, format='jpeg', bbox_inches='tight', dpi=300)

    print(f"Summary table saved to: {output_image_path}")

def filter_measurements(base_name):
    """
    Filter the CSV file to include only the specified columns, rename a column, and sort by Classification.

    Args:
        base_name (str): Base name of the input CSV file. The full path is constructed using this base name.

    Returns:
        pd.DataFrame: Filtered DataFrame with specified columns, renamed column, and sorted by Classification.
    """
    # Construct the input and output file paths using the base name
    # input_file = f"{QUPATH_PROJECT_DIR}/{base_name}_measurements.csv"
    input_file = os.path.join(TEMP_OUTPUTS_DIR, "Qupath_measurements", f"{base_name}_resized_measurements.csv")
    output_file = os.path.join(TEMP_OUTPUTS_DIR, "Qupath_measurements", f"{base_name}_resized_filtered_measurements.csv")

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Select the specified columns
    # filtered_df = df[['Object ID', 'Object type', 'Classification', 'ROI: 1.00 px per pixel: Channel 1: Mean']]
    filtered_df = df[['Object ID', 'Object type', 'Classification', 'ROI: 1.00 px per pixel: Brightness: Mean']]

    # Rename the column 'ROI: 1.00 px per pixel: Channel 1: Mean' to 'Mean Intensity'
    # filtered_df = filtered_df.rename(columns={'ROI: 1.00 px per pixel: Channel 1: Mean': 'Mean Intensity'})
    filtered_df = filtered_df.rename(columns={'ROI: 1.00 px per pixel: Brightness: Mean': 'Mean Intensity'})

    # Sort the DataFrame alphabetically by the 'Classification' column
    filtered_df = filtered_df.sort_values(by='Classification')

    # Save the filtered and sorted data to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered and sorted data saved to {output_file}")

    return filtered_df
