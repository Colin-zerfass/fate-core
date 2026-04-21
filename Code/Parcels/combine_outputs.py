import pandas as pd 

import os
import pandas as pd
import sys
import tomllib

def combine_outputs_csv(config):
    config = config 

    FileName = config['output_name']

    output_csv   = rf"./saved_output/{FileName}.csv"
    # --- SETTINGS ---
    input_folder = r"./output"      # folder containing your csv files
    #output_csv   = r"./saved_output/cmems_dynamical2023.csv"    # final combined file name

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


    with open(r'output\Output_logs.txt', 'a') as o: 
        o.write(f'Saved model run {FileName} \n')


if __name__ == '__main__': 
    configfile = sys.argv[1]
    with open(configfile, 'rb') as f:
        config = tomllib.load(f)
    
    combine_outputs_csv(config)
    
