import pandas as pd
import geopandas as gpd 
import numpy as np 
from functions.funcs import haversine_df, generate_longlist
from  functions.output_functions import merge_forecast_true, calc_iniial_lat, calc_intial_speed_dif, calc_projection_initial_angle

forecast1 = "cmems_dynamical_2022_2025_wind"
forecast2 = "OSCAR_2022_2025_wind"

# LOAD DATA 
#-----------------

## dFAD data
ds = gpd.read_parquet(r"..\Data\Mappedwinds_OSCAR_SAT_MI_Cleanedspeeds.parquet") 
ds = ds.reset_index(drop = True)
# load Forecast data to be merged
fc = pd.read_csv("saved_output/"+ forecast1 +".csv")
fc1 = pd.read_csv("saved_output/"+ forecast2 +".csv")

fc["Time"] = pd.to_datetime(fc["Time"])
fc['error_km'] = haversine_df(fc, "lat_true", "lon_true", "lat_forcast", "lon_forcast")
fc1["Time"] = pd.to_datetime(fc1["Time"])
fc1['error_km'] = haversine_df(fc1, "lat_true", "lon_true", "lat_forcast", "lon_forcast")

longlist  = generate_longlist(ds)

# combine dFAD and Forecast data 
#----------------------------------

oscar_merged = merge_forecast_true(fc1, longlist)
cmems_merged = merge_forecast_true(fc, longlist)
##remove duplicate forecasts start at the same time 
cmems_merged["starttime"] = cmems_merged['Time']- pd.to_timedelta(cmems_merged["leadtime"], unit = "hours").dt.round("min")
oscar_merged["starttime"] = oscar_merged['Time']- pd.to_timedelta(oscar_merged["leadtime"], unit = "hours").dt.round("min")

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

ia_save = ia_forecast[["BuoyID", "Time", "lat_true", "lon_true", "lon_forcast", "lat_forcast", "leadtime", "best_model", "initial_angle_used"]]
ia_save.to_csv(r"saved_output\intial_speed_dif_OSCAR_CMEMS_wind_2022_2025.csv")