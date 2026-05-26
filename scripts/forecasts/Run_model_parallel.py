
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
<<<<<<< HEAD
    chunks = [c for c in np.array_split(daterange, n_cores*1) if len(c) > 0]
=======
    chunks = [c for c in np.array_split(daterange, n_cores*3) if len(c) > 0]
>>>>>>> c97b5263a47fd5ed1f5ba82078362fd321b9ec1e

    inputs = [
        (chunk[0], chunk[-1], idx, config_name)
        for idx, chunk in enumerate(chunks)
    ]

    n_chunks = len(chunks)

    def run_pool(model_fn):
        with mp.Pool(processes=n_cores) as pool:
            async_results = [pool.apply_async(model_fn, args=inp) for inp in inputs]
            results = []
            for n, ar in enumerate(async_results, 1):
                results.append(ar.get())
                if n % 10 == 0:
                    print(f"{n}/{n_chunks} completed")
        return results

    if config['dynamical']:
        results = run_pool(Run_model_dynamical)
    else:
        results = run_pool(Run_model_static)


    ## combines ouputs into one csv
    combine_outputs_csv(config)    






