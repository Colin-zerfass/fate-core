
"""Method of running the model on given number of threads, one model runs on each thread sectioned by the monthrange above"""
import multiprocessing as mp
import sys
import tomli as tomllib
import pandas as pd
import numpy as np
from functions.parcels.forecast_model_dynamic import Run_model_dynamical, log
from functions.parcels.forecast_model_static import Run_model_static
from combine_outputs import combine_outputs_csv 
import functions.settings as settings

"""Method of running the model on given number of threads, one model runs on each thread sectioned by the monthrange above"""

if __name__ == '__main__':
    config_name = sys.argv[1]
    with open(config_name, 'rb') as f:
        config = tomllib.load(f)

    # clearing previous output files stored in scripts/forecasts/outputs
    outputpath = settings.CORE_DIR / 'scripts' / 'forecasts' / 'output'
    for output in outputpath.rglob('*.csv'):
        output.unlink()

    totalstartdate = config['startdate']
    totalenddate = config['enddate']
    n_cores = config['parallel_cores']
    print(totalstartdate)
    print(totalenddate)
    log(f"Starting Run: {config['output_name']}\n")

    # Split the full day range evenly across available cores.
    # Each worker gets a contiguous block of days rather than a calendar month.
    daterange = pd.date_range(totalstartdate, totalenddate, freq="D")
    chunks = [c for c in np.array_split(daterange, n_cores*1) if len(c) > 0]

    inputs = [
        (chunk[0], chunk[-1], idx, config_name)
        for idx, chunk in enumerate(chunks)
    ]

    if config['dynamical'] == True: 
        with mp.Pool(processes=n_cores) as pool:
            results = pool.starmap(Run_model_dynamical, inputs)

    if config['dynamical'] == False: 
        with mp.Pool(processes=n_cores) as pool:
            results = pool.starmap(Run_model_static, inputs)


    ## combines ouputs into one csv
    combine_outputs_csv(config)    






