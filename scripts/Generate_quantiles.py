import numpy as np 
import pandas as pd 
import geopandas as gpd
import functions.funcs as funcs 
import functions.output_functions as opf
import xarray as xr
import os
import functions.settings as settings
import tomllib
import sys 

"""Genrates Regression for forecast errors. Error(qauntile, leadtime, initial_speed_dif, latitude)

"""



def calc_quantile_regression(fc:pd.DataFrame, ds:gpd.GeoDataFrame, output_location):
    fc["Time"] = pd.to_datetime(fc["Time"])
    fc['error_km'] = funcs.haversine_df(fc, "lat_true", "lon_true", "lat_forcast", "lon_forcast")

    ## Unpacking True dFAD data into one list
    longlist = pd.DataFrame({})
    longlist["Time"] = funcs.Column_to_List(ds, "TimeStamp", idlist = False)
    longlist["lats"], longlist["lons"] = funcs.list_of_latlon(ds, False)
    longlist["x_speed"] = funcs.Column_to_List(ds, "x_speed", idlist = False)
    longlist["y_speed"] = funcs.Column_to_List(ds, "y_speed", idlist = False)
    longlist["v_mapped"], longlist["BuoyID"]  =funcs.Column_to_List(ds, "mapped_v", idlist = True)
    longlist["v_mapped_OSCAR"], longlist["BuoyID"]  =funcs.Column_to_List(ds, "mapped_v_oscar", idlist = True)
    longlist["u_mapped"] = funcs.Column_to_List(ds, "mapped_u", idlist = False)
    longlist["u_mapped_OSCAR"] = funcs.Column_to_List(ds, "mapped_u_oscar", idlist = False)
    longlist.Time = pd.to_datetime(longlist.Time)

    merged = opf.merge_forecast_true(fc, longlist)
    merged = opf.add_starttime(merged)
    merged["speed"] = np.sqrt(merged.x_speed**2 + merged.y_speed**2)
    merged = opf.calc_intial_speed_dif(merged)
    merged = opf.calc_iniial_lat(merged)

    output = opf.quantile_regression(merged)
    output.to_netcdf(output_location)
    print(f'saved qunatile regression to {output_location}')


if __name__ == '__main__':

    output_loc = settings.DATA_DIR / 'regression_quantiles_leadtimes_cmems_bias_pers_2026.nc'

    config_name = sys.argv[1]
    with open(config_name, 'rb') as f:
        config = tomllib.load(f)

    ds = gpd.read_parquet(settings.dFAD_DATA) 
    fc = pd.read_csv(settings.FORECAST_DIR / (config['output_name'] + '.csv'))

    calc_quantile_regression(fc, ds, output_loc)



