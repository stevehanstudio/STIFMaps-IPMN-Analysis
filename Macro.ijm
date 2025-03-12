// Define the input file
inputFile = "/home/steve/Projects/WeaverLab/STIFMaps/27620_STIFMap_stitched.png";

// Define the output file for the cropped image
outputFile = "/home/steve/Projects/WeaverLab/STIFMaps/27620_cropped.tif";

// Define the dimensions of the original DAPI image
origWidth = 25922;
origHeight = 31398;

// Define the total width and height for the stitched image
totalWidth = 5003 * 7;
totalHeight = 5003 * 6;

// Open the stitched image
open(inputFile);

// Resize the stitched image to the total width and height
run("Size...", "width=" + totalWidth + " height=" + totalHeight + " constrain=false");

// Calculate the crop region to match the original DAPI image dimensions
left = (totalWidth - origWidth) / 2;
top = (totalHeight - origHeight) / 2;
width = origWidth;
height = origHeight;

// Crop the resized image
makeRectangle(left, top, width, height);
run("Crop");

// Save the cropped image as a TIFF file
saveAs("Tiff", outputFile);

// Close the image
close();
