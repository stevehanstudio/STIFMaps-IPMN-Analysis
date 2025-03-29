// Get input arguments
inputPath = getArgument();
if (inputPath == "") {
    exit("No input file specified. Usage: ImageJ-linux64 --headless -macro extend_canvas.ijm input_image_path");
}

// Open the specified image
open(inputPath);

// Get image info and create output path
inputDir = File.getDirectory(inputPath);
inputName = File.getName(inputPath);
baseName = inputName;
if (indexOf(inputName, ".") > 0) {
    baseName = substring(inputName, 0, lastIndexOf(inputName, "."));
}
outputName = baseName + "_extend.tif";
outputPath = inputDir + outputName;

// Get current dimensions
width = getWidth();
height = getHeight();

// Calculate how much to extend
newHeight = width;  // Make height equal to width for square
extraHeight = newHeight - height;  // How much to add

// Only extend if width is greater than height
if (width > height) {
    // Extend canvas, placing the image at the top (adds black to bottom)
    run("Canvas Size...", "width=" + width + " height=" + newHeight + " position=Top-Center zero");
    print("Canvas extended with " + extraHeight + " pixels of black added to the bottom");
    
    // Save the modified image
    saveAs("Tiff", outputPath);
    print("Saved extended image as: " + outputPath);
} else {
    print("Image is already square or taller than wide. No extension needed.");
}

// Close the image and exit
close();
run("Quit");