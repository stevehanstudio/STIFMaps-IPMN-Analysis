from ij import IJ
import os

# PROJECT_DIR = "B:/Projects/WeaverLab/STIFMaps-IPMN-Analysis/"
print(os.getcwd())
PROJECT_DIR = os.path.join(os.getcwd(), 'Projects', 'WeaverLab', 'STIFMaps-IPMN-Analysis')
TEMP_OUTPUTS_DIR = os.path.join(PROJECT_DIR, "temp_outputs")
ORIG_IMAGE_DIR = os.path.join(PROJECT_DIR, "IPMN_images")

if not os.path.exists(TEMP_OUTPUTS_DIR):
    os.makedirs(TEMP_OUTPUTS_DIR)
    print("Created temp output directory: " + TEMP_OUTPUTS_DIR)

output_folder = os.path.join(TEMP_OUTPUTS_DIR, "resized0.25_IPMN_images")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print("Created resized image output directory: " + output_folder)

# List all TIFF images in the input folder
image_files = [f for f in os.listdir(ORIG_IMAGE_DIR) if f.endswith((".tiff", ".tif"))]
print("Found " + str(len(image_files)) + " image(s) to process.")

# Loop through each image
for image_name in image_files:
    print("Processing: " + image_name)

    # Open image
    imp = IJ.openImage(os.path.join(ORIG_IMAGE_DIR, image_name))

    if imp is None:
        print("Skipping " + image_name + ", failed to open.")
        continue  # Skip if the image couldn't be opened

    # Get original dimensions
    original_width = imp.getWidth()
    original_height = imp.getHeight()

    # Compute new dimensions (1/4th scale)
    new_width = original_width // 4
    new_height = original_height // 4

    # Resize the image
    imp = imp.resize(new_width, new_height, "bilinear")

    # Construct new filename with "_resized"
    # base_name = os.path.splitext(image_name)[0]  # Remove extension
    base_name, ext = os.path.splitext(os.path.basename(image_name))
    # new_filename = f"{base_name}_resized.tiff"
    new_filename = "{}_resized{}".format(base_name, ext)
    print("Saving resized image as: " + new_filename)

    # Save the resized image
    IJ.saveAs(imp, "Tiff", os.path.join(TEMP_OUTPUTS_DIR, "resized_IPMN_images", new_filename))

    # Close image
    imp.close()

print("Image resizing complete.")

