import os
import re
import numpy as np
import tifffile
from PIL import Image
import matplotlib.pyplot as plt

# Constants - Directories, Global Variables
PROJECT_DIR = os.getcwd()
ORIG_IMAGE_DIR = os.path.join(PROJECT_DIR, 'IPMN_images')
MODELS_DIR = os.path.join(PROJECT_DIR, '../STIFMap_dataset/trained_models')

TEMP_OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'temp_outputs')
FINAL_OUTPUTS_DIR = os.path.join(PROJECT_DIR, 'final_outputs')

# BASE_NAMES = ['1865']
BASE_NAMES = ['27620', '15806', '4601', '13401', '5114', '1865']
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
    
def generate_colormap_legend(base_index):
    """
    Generate and save a colormap legend for the raw stiffness values.

    Parameters:
    - base_index: Index to BASE_NAMES to determine the base name.
    """
    base_name = BASE_NAMES[base_index]
    stiffness_dir = os.path.join(TEMP_OUTPUTS_DIR, base_name, "STIFMap_tiles")

    # Initialize min and max values
    min_value = float('inf')
    max_value = float('-inf')

    # Find all .npy files in the directory
    for file_name in os.listdir(stiffness_dir):
        if file_name.endswith('.npy'):
            npy_path = os.path.join(stiffness_dir, file_name)
            stiffness_map = np.load(npy_path)

            # Update min and max values
            min_value = min(min_value, np.min(stiffness_map))
            max_value = max(max_value, np.max(stiffness_map))

    # Create a dummy image for the colormap
    fig, ax = plt.subplots(figsize=(2, 6))
    dummy_img = ax.imshow([[min_value, max_value]], cmap='viridis', aspect='auto')

    # Remove the dummy axes
    ax.remove()

    # Create colorbar with proper labeling
    cbar = fig.colorbar(dummy_img, ax=fig.add_subplot(111))
    cbar.set_label("Stiffness (Young's Modulus) [kPa]", fontsize=14)

    # Save the colormap legend
    legend_path = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}_stiffness_colormap_legend.png")
    plt.savefig(legend_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Colormap legend saved as {legend_path}")

def stitch_STIFMap_tiles(base_name_index, image_format='png'):
    # Dynamically determine the number of rows and columns
    base_name = BASE_NAMES[base_name_index]
    tile_pattern = re.compile(rf"{base_name}_(\d+)_(\d+)\.{image_format}")
    STIFMaps_directory = os.path.join(TEMP_OUTPUTS_DIR, base_name, "STIFMap_tiles")
    output_filename = os.path.join(FINAL_OUTPUTS_DIR, f"{base_name}__STIFMap_stitched.png")
    
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

    # Determine image dimensions from the first available .png tile
    image_width = image_height = None
    for row in range(num_rows):
        for col in range(num_cols):
            image_filename = f"{base_name}_{row}_{col}.{image_format}"
            image_path = os.path.join(STIFMaps_directory, image_filename)
            if os.path.exists(image_path):
                try:
                    with Image.open(image_path) as image:
                        image_width, image_height = image.size
                    break
                except Exception as e:
                    print(f"Error opening {image_path}: {e}")
        if image_width is not None:
            break

    if image_width is None:
        raise ValueError("No valid .png image files found to determine dimensions.")

    stitched_width = num_cols * image_width
    stitched_height = num_rows * image_height
    stitched_image = Image.new('RGB', (stitched_width, stitched_height), color='white')

    for row in range(num_rows):
        for col in range(num_cols):
            image_filename = f"{base_name}_{row}_{col}.{image_format}"
            image_path = os.path.join(STIFMaps_directory, image_filename)

            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path)
                except Exception as e:
                    print(f"Error opening {image_path}: {e}")
                    image = Image.new('RGB', (image_width, image_height), color='white')
                    print(f"Missing tile: {image_filename}. Replacing with a white tile.")
            else:
                image = Image.new('RGB', (image_width, image_height), color='white')
                print(f"Missing tile: {image_filename}. Replacing with a white tile.")

            x = col * image_width
            y = row * image_height
            stitched_image.paste(image, (x, y))

    stitched_image.save(output_filename)
    print(f"Stitched image saved as {output_filename}")

# def save_stiffness_colormap_legend(
#     cmap="viridis",
#     min_value=0,
#     max_value=54286,
#     label="Stiffness (Young's Modulus) [kPa]",
#     save_path="stiffness_color_map_legend.png"
# ):
#     fig, ax = plt.subplots(figsize=(2, 6))  # Tall, narrow figure
    
#     # Create a dummy colormap
#     gradient = np.linspace(0, 1, 256).reshape(-1, 1)
#     ax.imshow(gradient, aspect='auto', cmap=cmap)
    
#     # Hide axes
#     ax.set_axis_off()
    
#     # Add colorbar
#     norm = plt.Normalize(vmin=min_value, vmax=max_value)
#     cbar = fig.colorbar(
#         plt.cm.ScalarMappable(norm=norm, cmap=cmap),
#         ax=ax,
#         orientation='vertical',
#         fraction=0.1,
#         pad=0.05
#     )
#     cbar.set_label(label, fontsize=14, rotation=90)
    
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False)
#     plt.close()
#     print(f"Legend saved to {save_path}")

# def save_stiffness_colormap(stitched_image_path, base_name, output_dir=FINAL_OUTPUTS_DIR):
#     """
#     Converts the stitched image to a normalized grayscale colormap and saves it as {base_name}_stiffness.png.
    
#     Args:
#         stitched_image_path (str): Path to the stitched image (RGB).
#         base_name (str): Base name for naming the output.
#         output_dir (str): Directory where the stiffness colormap will be saved.
#     """
#     # Load stitched image and convert to grayscale
#     stitched_rgb = Image.open(stitched_image_path)
#     stitched_gray = stitched_rgb.convert("L")
#     stitched_array = np.array(stitched_gray, dtype=np.float32)

#     # Normalize the grayscale array
#     stitched_array_normalized = (stitched_array - np.min(stitched_array)) / (np.max(stitched_array) - np.min(stitched_array))

#     # Save as color-mapped PNG
#     stiffness_output_path = os.path.join(output_dir, f"{base_name}_stiffness.png")
#     plt.imsave(stiffness_output_path, stitched_array_normalized, cmap="viridis")
#     print(f"Stiffness colormap saved as {stiffness_output_path}")
