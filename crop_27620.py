from PIL import Image
import os

# Define the path to the image file
image_path = os.path.join(os.getcwd(), 'final_outputs', '27620_STIFMap_stitched.png')

# Load the original image
original_image = Image.open(image_path)

# Define the dimensions for cropping
cropped_width = 2209
cropped_height = 1824

# Crop the image from the top-left corner
cropped_image = original_image.crop((0, 0, cropped_width, cropped_height))

# Convert the cropped image to grayscale
gray_image = cropped_image.convert("L")

# Define the path and name for the saved image
save_path = os.path.join(os.getcwd(), 'final_outputs', '27620_STIFMap_stitched_cropped_gray.png')

# Save the grayscale image as PNG
gray_image.save(save_path)

print(f"Grayscale image saved to {save_path}")
