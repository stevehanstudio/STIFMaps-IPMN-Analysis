import json
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import shape, mapping, Polygon
from shapely.affinity import scale

# Scaling factors for width and height
xfact = 2209 / 31398  # Approximately 0.0704
yfact = 1824 / 25922  # Approximately 0.0704

# File paths
input_geojson_path = "C:/Users/steve/Projects/WeaverLab/analysis_panel_1/27620_annotations.geojson"
output_geojson_path = "C:/Users/steve/Projects/WeaverLab/analysis_panel_1/27620_scaled_annotations.geojson"
image_path = "C:/Users/steve/Projects/WeaverLab/STIFMaps-IPMN-Analysis/final_outputs/27620_STIFMap_stitched_cropped_gray.png"
output_csv_path = "C:/Users/steve/Projects/WeaverLab/analysis_panel_1/mean_intensity_results.csv"

# Load the original GeoJSON file
with open(input_geojson_path, 'r') as f:
    geojson_data = json.load(f)

# Debugging: Print entire GeoJSON structure
# print("Loaded GeoJSON Data:")
# print(json.dumps(geojson_data, indent=2))

# Debugging: Print a summary of the GeoJSON structure
print(f"Number of features in GeoJSON: {len(geojson_data['features'])}")

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
output_image_path = "C:/Users/steve/Projects/WeaverLab/analysis_panel_1/summary_table.jpeg"

# Read the generated CSV into a DataFrame
summary_df = pd.read_csv(output_csv_path)

# Summarize the data by classification
summary = df.groupby('Classification').agg(
    Total_Number=('Classification', 'size'),
    Mean_Intensity=('Mean Intensity', 'mean')
).reset_index()

# File path for the summary table image
output_image_path = "C:/Users/steve/Projects/WeaverLab/analysis_panel_1/summary_table.jpeg"

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
