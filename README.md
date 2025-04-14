# STIFMaps for IPMN Tissue Analysis

This project is built on the [Spatially Transformed Inferential Force Maps (STIFMaps) project (by cstashko)](https://github.com/cstashko/STIFMaps) and is being developed as part of an internship at the Weaver Lab at UCSF during the Spring semester of 2025. The ultimate aim of the project is to enable earlier detection of pancreatic cancer through tissue stiffness analysis. This is a work in progress.

## Contents

-  [Project Overview](#project-overview)
-  [Repo Contents](#repo-contents)
-  [System Requirements](#system-requirements)
-  [Installation Guide](#installation-guide)
-  [Getting Started](#getting-started)
-  [Directories](#directories)
-  [License](#license)
-  [Acknowledgements](#acknowledgements)
-  [Contact](#contact)

## Project Overview

This project utilizes the STIFMap deep learning model to analyze DAPI and CNA35 stained images of pancreatic IPMN tissues. The process involves breaking down whole slide images into tiles, which are then analyzed by the model. The resulting predictions are used to classify and differentiate the tissue grade:

1. Low Grade (LG)
2. High Grade (HG)
3. Pancreatic Ductal Adenocarcinoma (PDAC)

The workflow involves:

-  Processing DAPI and CNA35 stained tissue images
-  Breaking down whole slide images into manageable tiles
-  Analyzing these tiles using pre-trained STIFMap models
-  Generating stiffness predictions for tissue classification
-  Post-processing results with tools like QuPath for annotation and analysis

## Workflow Diagram

![Workflow Diagram](https://github.com/cstashko/STIFMaps/blob/main/workflow.png)

## Repo Contents

-  `globals_and_helpers.py`: This file contains global variables and helper functions used throughout the project, such as file paths, model parameters, and utility functions for image processing and data handling.
-  `preprocess.ipynb`: This Jupyter Notebook focuses on the preprocessing steps required for the IPMN tissue images. It includes steps for image resizing, stain normalization, and tile extraction.
-  `gen_STIFMaps.ipynb`: This Jupyter Notebook implements the STIFMap model to the preprocessed tiles of IPMN tissues to predict the grade of tissue.
-  `../STIFMaps_dataset/trained_models/`: This directory should contain the pre-trained STIFMap models. Ensure the models are located here for the notebooks to function correctly. You can download the STIFMap model from [https://data.mendeley.com/datasets/vw2bb5jy99/2](https://data.mendeley.com/datasets/vw2bb5jy99/2).

## System Requirements

STIFMaps should run on any standard computer capable of running Jupyter and PyTorch. A minimum of 16 GB of RAM is recommended, especially when utilizing CUDA for GPU acceleration. The memory requirements will depend on the size of the whole slide images being processed. Downsampling images before stiffness prediction can help reduce memory consumption.

## Installation Guide

It is highly recommended to set up a dedicated virtual environment for this project. Follow these steps to install the necessary dependencies using conda:

1. **Create a conda environment:**

   ```
   conda create -n STIFMaps python=3.10
   ```

2. **Activate the environment:**

   ```
   conda activate STIFMaps
   ```

3. **Install pip:**

   ```
   conda install -n STIFMaps pip
   ```

4. **Install the necessary packages:**

   ```
   python3 -m pip install numpy pandas scikit-image matplotlib pytorch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html
   ```

## Getting Started

1. **Ensure Trained Models are Available:**

   -  Verify that the pre-trained STIFMap models are located in the `../STIFMaps_dataset/trained_models/` directory.
      -  You can download the STIFMap model from [https://data.mendeley.com/datasets/vw2bb5jy99/2](https://data.

## References

### Research Papers

### Tools

-  [https://github.com/cstashko/STIFMaps](https://github.com/cstashko/STIFMaps)
-  [PlantUML in VS Code, installation (Youtube)](https://www.youtube.com/watch?v=Tj41ggVdTNk)
-  [Adoptium.net](https://adoptium.net/): OpenJDK - used to install PlantUML locally to integrate with VS code
-  [Plant UML](https://plantuml.com/): Used to generate UML diagrams
