import pandas as pd 

import os
import pandas as pd
import sys
import tomllib
import functions.settings as settings 
from functions.parcels.forecast_model_dynamic import log

def combine_outputs_csv(config):
    config = config 

    FileName = config['output_name'] +'.csv'

    output_csv   = settings.FORECAST_DIR / FileName
    # --- SETTINGS ---
    input_folder = settings.CORE_DIR / 'scripts/forecasts/output'      # folder containing csv files
    #output_csv   = r"./saved_output/cmems_dynamical2023.csv"    # final combined file name

    # --- COLLECT ALL CSV FILE PATHS ---
    csv_files = [os.path.join(input_folder, f)
                for f in os.listdir(input_folder)
                if f.lower().endswith(".csv")]

    print(f"Found {len(csv_files)} CSV files.")

    # --- READ AND CONCATENATE ---
    df_list = []

    for file in csv_files:
        df = pd.read_csv(file)
        # Optional: add filename as a column
        # df["source_file"] = os.path.basename(file)
        df_list.append(df)

    combined = pd.concat(df_list, ignore_index=True)

    # --- SAVE ---
    combined.to_csv(output_csv, index=False)

    print(f"Combined CSV saved as: {output_csv}")

    log(f'Saved model run {FileName} \n')


if __name__ == '__main__': 
    configfile = sys.argv[1]
    with open(configfile, 'rb') as f:
        config = tomllib.load(f)
    
    combine_outputs_csv(config)
    
