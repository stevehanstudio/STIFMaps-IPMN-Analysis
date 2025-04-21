import os
import pandas as pd
from globals_and_helpers import (
    BASE_NAMES,
    TEMP_OUTPUTS_DIR,
    # filter_measurements,
)

# for base_name in BASE_NAMES:
#     try:
#         filter_measurements(base_name)
#     except Exception as e:
#         print(f"Error processing {base_name}: {e}")

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
    # input_file = os.path.join(TEMP_OUTPUTS_DIR, "Qupath_measurements", f"{base_name}_resized_measurements.csv")
    # output_file = os.path.join(TEMP_OUTPUTS_DIR, "Qupath_measurements", f"{base_name}_resized_filtered_measurements.csv")
    input_file = os.path.join(TEMP_OUTPUTS_DIR, "Qupath_measurements", f"{base_name}_measurements_with_tiling.csv")
    output_file = os.path.join(TEMP_OUTPUTS_DIR, "Qupath_measurements", f"{base_name}_filtered_measurements_with_tiling.csv")

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Select the specified columns
    filtered_df = df[['Object ID', 'Object type', 'Classification', 'ROI: 1.00 px per pixel: Channel 1: Mean']]
    # filtered_df = df[['Object ID', 'Object type', 'Classification', 'ROI: 1.00 px per pixel: Brightness: Mean']]

    # Rename the column 'ROI: 1.00 px per pixel: Channel 1: Mean' to 'Mean Intensity'
    filtered_df = filtered_df.rename(columns={'ROI: 1.00 px per pixel: Channel 1: Mean': 'Mean Intensity'})
    # filtered_df = filtered_df.rename(columns={'ROI: 1.00 px per pixel: Brightness: Mean': 'Mean Intensity'})

    # Sort the DataFrame alphabetically by the 'Classification' column
    filtered_df = filtered_df.sort_values(by='Classification')

    # Save the filtered and sorted data to a new CSV file
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered and sorted data saved to {output_file}")

    return filtered_df

filter_measurements("27620")
# filter_measurements("4601")
# filter_measurements("7002")

