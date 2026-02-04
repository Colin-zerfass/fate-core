import pandas as pd 

import os
import pandas as pd

# --- SETTINGS ---
input_folder = r"./output"      # folder containing your csv files
output_csv   = r"./saved_output/cmems_dynamicalv1.csv"    # final combined file name

# --- COLLECT ALL CSV FILE PATHS ---
csv_files = [os.path.join(input_folder, f)
             for f in os.listdir(input_folder)
             if f.lower().endswith(".csv")]

print(f"Found {len(csv_files)} CSV files.")

# --- READ AND CONCATENATE ---
df_list = []

for file in csv_files:
    print(f"Reading: {file}")
    df = pd.read_csv(file)
    # Optional: add filename as a column
    # df["source_file"] = os.path.basename(file)
    df_list.append(df)

combined = pd.concat(df_list, ignore_index=True)

# --- SAVE ---
combined.to_csv(output_csv, index=False)

print(f"Combined CSV saved as: {output_csv}")
