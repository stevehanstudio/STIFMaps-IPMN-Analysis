// Define the input file
inputFile = "/home/steve/Projects/WeaverLab/STIFMaps/STIFMap_normalized_images_v1/27620_STIFMap_stitched_v1.png";
outputFile = "/home/steve/Projects/WeaverLab/STIFMaps/final_outputs/27620_final.tif";
legendFile = "/home/steve/Projects/WeaverLab/STIFMaps/stiffness_color_map_legend.png";
origWidth = 25922;
origHeight = 31398;
totalWidth = 5003 * 7;
totalHeight = 5003 * 6;

// Step 1: Open and resize the STIFMap
print("Step 1: Opening stitched image");
open(inputFile);
run("Scale...", "x=" + totalWidth/getWidth() + " y=" + totalHeight/getHeight() + " interpolation=Bilinear create");
rename("Resized");
resizedID = getImageID();
selectWindow("27620_STIFMap_stitched_v1.png");
close();

// Step 2: Crop from left/top to origWidth x origHeight
print("Step 2: Cropping to original dimensions (from left/top)");
selectImage(resizedID);
makeRectangle(0, 0, origWidth, origHeight);
run("Crop");
rename("Cropped");
croppedID = getImageID();

/* Commenting out legend processing since we'll do it manually
// Step 3: Process the legend
print("Step 3: Processing legend");
open(legendFile);
legendID = getImageID();
// Calculate legend size (1/5 height of main image)
legendHeight = origHeight / 5.0;
legendScale = legendHeight / getHeight();
legendWidth = getWidth() * legendScale;
run("Scale...", "x=" + legendScale + " y=" + legendScale + " interpolation=Bilinear create");
rename("ScaledLegend");
scaledLegendID = getImageID();
selectWindow(legendFile);
close();
*/

// Calculate legend dimensions for canvas size (still needed even though we'll add manually)
print("Step 3: Calculating legend dimensions");
// Temporarily open legend to get dimensions
open(legendFile);
legendWidth = getWidth();
legendHeight = getHeight();
close();
// Calculate scaled dimensions
scaledLegendHeight = origHeight / 5.0;
scaledLegendWidth = legendWidth * (scaledLegendHeight / legendHeight);

// Step 4: Expand canvas on right side only
print("Step 4: Expanding canvas on right side");
selectImage(croppedID);
padding = 20; // Padding around legend
newWidth = origWidth + scaledLegendWidth + padding;
// To expand only on the right, use position=Left
run("Canvas Size...", "width=" + newWidth + " height=" + origHeight + " position=Left-Center zero");

/* Commenting out legend pasting since we'll do it manually
// Step 5: Add legend to upper right corner
print("Step 5: Adding legend to upper right corner");
selectImage(scaledLegendID);
run("Select All");
run("Copy");
close();

selectImage(croppedID);
// Position for upper right corner
legendX = origWidth + padding/2;
legendY = padding/2;
// Make a selection where the legend should go
makeRectangle(legendX, legendY, legendWidth, legendHeight);
run("Paste");
run("Select None");
*/

// Step 5: Save the image with expanded canvas (without legend)
print("Step 5: Saving final image (without legend)");
saveAs("Tiff", outputFile);

// Print instruction for manual legend addition
//print("\n*** MANUAL LEGEND ADDITION INSTRUCTIONS ***");
//print("Legend should be placed at X=" + (origWidth + padding/2) + ", Y=" + (padding/2));
//print("Scaled legend dimensions should be: Width=" + scaledLegendWidth + ", Height=" + scaledLegendHeight);

//print("Done!");