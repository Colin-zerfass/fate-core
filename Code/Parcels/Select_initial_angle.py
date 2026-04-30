import pandas as pd
import geopandas as gpd 
import numpy as np 
import xarray as xr
from functions.funcs import haversine_df, generate_longlist
from  functions.output_functions import merge_forecast_true, calc_iniial_lat, calc_intial_speed_dif, calc_projection_initial_angle

forecast1 = r"Final\OSCAR_wind_pers_2026"
forecast2 = r"Final\cmems_wind_pers_2026"

forecast1_regression = r"..\Data\regression_quantiles_leadtimes_oscar_wind_pers_2026.nc"
forecast2_regression = r"..\Data\regression_quantiles_leadtimes_cmems_wind_pers_2026.nc"


output = r'Final\Initial_angle_v1_2026'

leadtime = 6*24 # leadtime to select lowest perdicted error at. 

# LOAD DATA 
#-----------------
## dFAD data
ds = gpd.read_parquet(r"..\Data\SAT_MI_FAD_cleanedspeeds_2026-01-01_mapped_all.parquet") 
ds = ds.reset_index(drop = True)
# load Forecast data to be merged
fc = pd.read_csv("saved_output/"+ forecast1 +".csv")
fc1 = pd.read_csv("saved_output/"+ forecast2 +".csv")

reg = xr.open_dataset(forecast1_regression)
reg1 = xr.open_dataset(forecast2_regression)

fc["Time"] = pd.to_datetime(fc["Time"])
fc['error_km'] = haversine_df(fc, "lat_true", "lon_true", "lat_forcast", "lon_forcast")
fc1["Time"] = pd.to_datetime(fc1["Time"])
fc1['error_km'] = haversine_df(fc1, "lat_true", "lon_true", "lat_forcast", "lon_forcast")

longlist  = generate_longlist(ds, extra_columns= ['mapped_u', 'mapped_u_oscar', 'mapped_v', 'mapped_v_oscar'])
longlist['Time'] = pd.to_datetime(longlist['Time'])

## rename longlist to match calc Projection initial angle and speed diff functions
#-----------------------------------------------------------------------------------
longlist = longlist.rename(columns = {
    'mapped_u' : 'u_mapped', 'mapped_v': 'v_mapped',
    'mapped_u_oscar': 'u_mapped_OSCAR', 'mapped_v_oscar' : 'v_mapped_OSCAR'})

# combine dFAD and Forecast data 
#----------------------------------
oscar_merged = merge_forecast_true(fc, longlist)
cmems_merged = merge_forecast_true(fc1, longlist)
##remove duplicate forecasts start at the same time 
cmems_merged["starttime"] = cmems_merged['Time']- pd.to_timedelta(cmems_merged["leadtime"], unit = "hours").dt.round("min")
oscar_merged["starttime"] = oscar_merged['Time']- pd.to_timedelta(oscar_merged["leadtime"], unit = "hours").dt.round("min")

print(f'Length of merged CMEMs, OSCAR { len(cmems_merged), len(oscar_merged)}')
print(oscar_merged.columns)


# Cal Speed differance
#-------------------------
## calcuate inital angle, and speed differance for both CMEMs and OSCAR
oscar_merged = calc_projection_initial_angle(oscar_merged, "_OSCAR")
cmems_merged = calc_projection_initial_angle(cmems_merged)
oscar_merged = calc_projection_initial_angle(oscar_merged)
cmems_merged = calc_projection_initial_angle(cmems_merged, "_OSCAR")
oscar_merged = calc_intial_speed_dif(oscar_merged, "_OSCAR")
cmems_merged = calc_intial_speed_dif(cmems_merged)
oscar_merged = calc_intial_speed_dif(oscar_merged)
cmems_merged = calc_intial_speed_dif(cmems_merged, "_OSCAR")
print('merged dFAd and Forecasts, \ncalculated Init_speed_dif')

# Select between models  
#-------------------------
def lower_speed_dif(group):
    if group.empty:
        return group
    
    idx = group["leadtime"].idxmin()
    cmemsi = group.at[idx, "initial_speed_dif_mag"]
    oscari = group.at[idx, "initial_speed_dif_mag_OSCAR"]

    choice = "cmems" if cmemsi < oscari else "OSCAR"

    g = group.copy()
    g["best_model"] = choice
    return g

##updating to not just pick lower speed dif but decided based on what has the lower expected angle based on regression
""" 1) load in both regresssions
    2) Calc expected error of the regressions 
    3) Compair and pick forecast with lower regression
"""
#2 calc expected error from the regresssion
def expected_error(merged, reg:xr.Dataset, leadtime = 72, q = 0.5, sufix = None):
    regvalues = reg.sel(leadtime = leadtime, q = q, method = 'nearest')
    if sufix is not None:
        initial_lat = 'initial_lat'  # lat is observation-based, not model-dependent
        initial_speed_dif_mag = 'initial_speed_dif_mag' + sufix
    else:
        initial_lat = 'initial_lat' 
        initial_speed_dif_mag = 'initial_speed_dif_mag'

    ilat = regvalues.initial_lat.values
    ispeed = regvalues.initial_speed_dif_mag.values
    intercept = regvalues.Intercept.values
    output_col = 'expected_error' if sufix is None else 'expected_error' + sufix
    merged[output_col] = merged[initial_lat]*ilat + merged[initial_speed_dif_mag]*ispeed + intercept
    return merged

oscar_merged = calc_iniial_lat(oscar_merged)
cmems_merged = calc_iniial_lat(cmems_merged)
oscar_merged = expected_error(oscar_merged, reg, leadtime = leadtime, q = 0.5, sufix = '_OSCAR')
oscar_merged = expected_error(oscar_merged, reg1, leadtime = leadtime, q = 0.5)
cmems_merged = expected_error(cmems_merged, reg1, leadtime = leadtime, q = 0.5)
cmems_merged = expected_error(cmems_merged, reg, leadtime = leadtime, q = 0.5, sufix = '_OSCAR')


def lower_expected_error(group):
    if group.empty:
        return group
    
    idx = group["leadtime"].idxmin()
    cmemsi = group.at[idx, "expected_error"]
    oscari = group.at[idx, "expected_error_OSCAR"]

    choice = "cmems" if cmemsi < oscari else "OSCAR"

    g = group.copy()
    g["best_model"] = choice
    return g

cmems_grouped = cmems_merged.groupby(["BuoyID", "starttime"]).apply(lower_speed_dif, include_groups=False)
OSCAR_grouped = oscar_merged.groupby(["BuoyID", "starttime"]).apply(lower_speed_dif, include_groups=False)
print('Selected best models, now recombining')
cmems_forecasts = cmems_grouped.reset_index().query("best_model == 'cmems'")
oscar_forecasts = OSCAR_grouped.reset_index().query("best_model == 'OSCAR'")
oscar_forecasts["initial_angle_used"] = oscar_forecasts["initial_angle_OSCAR"]
cmems_forecasts["initial_angle_used"] = cmems_forecasts["initial_angle"]
oscar_forecasts["projection_used"] = oscar_forecasts["projection_OSCAR"]
cmems_forecasts["projection_used"] = cmems_forecasts["projection"]
print('recombined, now saving')

# Recombine and Save forecasts 
#-------------------------------
ia_forecast = pd.concat([oscar_forecasts,cmems_forecasts])
ia_save = ia_forecast[["BuoyID", "Time", "lat_true", "lon_true", 
                       "lon_forcast", "lat_forcast", "leadtime", "best_model", "initial_angle_used"]]
ia_save.to_csv("saved_output/" + output + ".csv")