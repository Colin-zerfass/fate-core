### Make no Forecasts 

import pandas as pd 
import functions.settings as settings 
import functions.output_functions as opf
import tomllib 
import sys


def No_forecast(fc:pd.DataFrame):
    fc = opf.add_starttime(fc)
    fc = opf.calc_initial_lat_lon(fc)
    fc = fc.sort_values('leadtime').reset_index(drop=True)
    fc['lat_forcast'] = fc['lati']
    fc['lon_forcast'] = fc['loni']
    return fc[['BuoyID','Time','lat_true','lon_true','lat_forcast','lon_forcast','leadtime']]

if __name__ == '__main__':
    # load the config file of actual forecast that were made   
    config_name = sys.argv[1]
    with open(config_name, 'rb') as f:
        config = tomllib.load(f)

    ds = pd.read_csv(settings.FORECAST_DIR  / (config['output_name'] +'.csv'))

    no_forecast = No_forecast(ds)
    no_forecast.to_csv(settings.FORECAST_DIR/ 'No_forecast.csv')

