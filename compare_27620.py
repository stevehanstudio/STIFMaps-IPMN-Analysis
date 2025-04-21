from globals_and_helpers import (
    PROJECT_DIR,
    TEMP_OUTPUTS_DIR,
    FINAL_OUTPUTS_DIR,
 )
import os
import pandas as pd

datapath = os.path.join(PROJECT_DIR, 'final_outputs', '0.25 scaling v. tiling')

# Load the CSV files into dataframes
df1 = pd.read_csv(os.path.join(datapath, '27620_0.25scale.csv'))
df2 = pd.read_csv(os.path.join(datapath, '27620_with_tiling.csv'))

# Filter the dataframes to keep only the specified columns
columns_to_keep = ['Object ID', 'Classification', 'Stiffness (log (E actual) PA)']
df1 = df1[columns_to_keep]
df2 = df2[columns_to_keep]

print(df1.describe())
print(df2.describe())

# Check if 'Object ID' column exists in both dataframes
if 'Object ID' in df1.columns and 'Object ID' in df2.columns:
    # Find Object IDs that are in both dataframes
    common_object_ids = pd.merge(df1['Object ID'], df2['Object ID'], how='inner')

    # Check if all Object IDs in df1 are in df2 and vice versa
    all_ids_match = set(df1['Object ID']) == set(df2['Object ID'])
else:
    common_object_ids = None
    all_ids_match = False

# Output the results
# print(common_object_ids, all_ids_match)
