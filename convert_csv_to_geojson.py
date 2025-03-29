import pandas as pd
import json
import math

# Load the annotations CSV
annotations_path = r'C:\Users\steve\Projects\WeaverLab\analysis_panel_1\27620_annotations.csv'
annotations = pd.read_csv(annotations_path)

features = []

for _, row in annotations.iterrows():
    try:
        # Handle ROI based on its type
        roi_type = row['ROI']
        classification = row['Classification']
        name = row['Name']
        
        if roi_type.startswith('Polygon'):  # Example: "Polygon" type
            # Assuming it contains coordinates in some standard format
            polygon_coords = eval(roi_type.split(':', 1)[-1])  # Parse coordinates if part of string
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [polygon_coords]  # GeoJSON expects a list of lists
                },
                "properties": {
                    "Classification": classification,
                    "Name": name
                }
            }
            features.append(feature)
        
        elif roi_type.startswith('Ellipse'):
            # Example: Generate an approximate polygon for the ellipse
            center_x, center_y = row['Centroid X µm'], row['Centroid Y µm']
            width, height = row['Area µm^2'], row['Perimeter µm']  # Replace with actual ellipse params
            num_points = 36  # Approximate with 36 points
            
            ellipse_coords = [
                [
                    center_x + (width / 2) * math.cos(angle),
                    center_y + (height / 2) * math.sin(angle)
                ]
                for angle in [2 * math.pi * i / num_points for i in range(num_points)]
            ]
            ellipse_coords.append(ellipse_coords[0])  # Close the shape
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [ellipse_coords]
                },
                "properties": {
                    "Classification": classification,
                    "Name": name
                }
            }
            features.append(feature)
        
        elif roi_type.startswith('Rectangle'):
            # Example: Parse rectangle bounds and create a polygon
            # Replace with actual rectangle parsing logic
            x1, y1, x2, y2 = [float(v) for v in roi_type.split(',')[1:5]]  # Example format
            rect_coords = [
                [x1, y1], [x2, y1], [x2, y2], [x1, y2], [x1, y1]  # Close the rectangle
            ]
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [rect_coords]
                },
                "properties": {
                    "Classification": classification,
                    "Name": name
                }
            }
            features.append(feature)
        
        else:
            print(f"Unhandled ROI type for Object ID {row['Object ID']}: {roi_type}")
    
    except Exception as e:
        print(f"Error processing ROI for Object ID {row['Object ID']}: {e}")

# Create GeoJSON
geojson = {
    "type": "FeatureCollection",
    "features": features
}

geojson_path = r'C:\Users\steve\Projects\WeaverLab\analysis_panel_1\27620_scaled_annotations.geojson'
with open(geojson_path, 'w') as f:
    json.dump(geojson, f)

print(f"GeoJSON file saved to {geojson_path}")
