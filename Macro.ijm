// Define the input file
inputFile = "/home/steve/Projects/WeaverLab/STIFMaps/STIFMap_normalized_images_v1/27620_STIFMap_stitched_v1.png";
outputFile = "/home/steve/Projects/WeaverLab/STIFMaps/final_outputs/27620_resized.tif";
totalWidth = 5003 * 7;  // 35021
totalHeight = 5003 * 6;  // 30018

// Step 1: Open the STIFMap
print("Step 1: Opening stitched image");
open(inputFile);

// Step 2: Resize the STIFMap
print("Step 2: Resizing stitched image");
originalWidth = getWidth();
originalHeight = getHeight();
print("Original dimensions: " + originalWidth + " x " + originalHeight);
print("Target dimensions: " + totalWidth + " x " + totalHeight);

// Calculate scaling factors
xScale = totalWidth / originalWidth;
yScale = totalHeight / originalHeight;
print("Scaling factors: " + xScale + " x " + yScale);

// Perform the resize
run("Scale...", "x=" + xScale + " y=" + yScale + " interpolation=Bilinear create");
print("New dimensions: " + getWidth() + " x " + getHeight());

// Step 3: Save the resized image
print("Step 3: Saving resized image");
saveAs("Tiff", outputFile);
close();

// Close original
selectWindow(File.getName(inputFile));
close();

print("Done! Resized image saved to: " + outputFile);