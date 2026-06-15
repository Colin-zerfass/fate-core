"""
Goal: Combine drifters into the same dataformat as the main dFAD ds. 
parquet where each trajecotry is its own row. Locations are stored as polylines
1) Columns needed. BuoyName, MinOfTimes, MaxOfTimes, TimeStamp, geometry, 
x_deg, y_deg, x_km, y_km, xy_km, x_speed, y_speed, xy_speed
"""

import pandas as pd
from pathlib import Path
import geopandas as gpd
import numpy as np
import shapely as sp 

import functions.funcs as funcs
import functions.Cleaning_functions as cleaning
import functions.settings as settings

def rename_columns(data:pd.DataFrame):
    return data.rename(columns={'CommID': 'BuoyName', 'DateTime': 'Timestamp'})

def combine_all_drifter_files(dir = settings.DATA_DIR / 'Drifters'):
    files = list(dir.glob('*.csv'))
    header = pd.read_csv(files[0]).columns
    output = pd.DataFrame(columns= header)
    print(f'Found {len(files)} files')

    loaded_datasets = []
    for file in files:
        a = pd.read_csv(file)
        loaded_datasets.append(a)

    output = pd.concat(loaded_datasets)
    output = rename_columns(output)
    output.to_csv(settings.DATA_DIR / 'Drifters_2026_06.csv')
    return output


def csv_parquet(data:pd.DataFrame, 
                outout_location: Path = settings.DATA_DIR / 'Drifter_cleaned_2026_06.parquet', 
                add_min_max_times = True, Geofenced_box= False) -> gpd.GeoDataFrame:
    data = data.copy()
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data['Longitude'] = data['Longitude'] - 360
    if add_min_max_times: 
        data["MinOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("min")
        data["MaxOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("max")

    if Geofenced_box:
        data = data.rename(columns= {'Latitude': 'lats', 'Longitude': 'lons', 'Timestamp': 'Time'})
        data = funcs.Query_longlist(dFADs_ds= data, 
                                    Time1 = '2022-01-01' , Time2 = '2026-01-01', 
                                    lat = [4.5, 7.75], lon = [-163.75, -160.6667])
        data = data.rename(columns = {'lats' : 'Latitude' , 'lons' : 'Longitude', 'Time': 'Timestamp'})
    Latitude_list = data.groupby("BuoyName")["Latitude"].apply(list)
    Longitude_list = data.groupby("BuoyName")["Longitude"].apply(list)
    times_list = data.groupby("BuoyName")["Timestamp"].apply(list)
    date_enter = data.groupby("BuoyName")["MinOfTimes"].apply(np.minimum.reduce)
    date_exit = data.groupby("BuoyName")["MaxOfTimes"].apply(np.minimum.reduce)
    BuoyNames = data["BuoyName"].unique()

    #makes Geometry lines from Lat, lons 
    lines = []
    for n in range(len(Latitude_list)):
        line = sp.linestrings(Longitude_list.iloc[n],Latitude_list.iloc[n])
        lines.append(line)

    newdata = gpd.GeoDataFrame(
        {"BuoyName":BuoyNames, "MinOfDate":date_enter, 
         "MaxOfDate": date_exit, "TimeStamp": times_list,
        "geometry": lines}, index= None)
    newdata = newdata.reset_index(drop = True)
    data = newdata
    data, _delx, _dely = funcs.add_distance_collumns(data)
    data = funcs.add_delta_time_collums(data)
    data = cleaning.Add_x_y_speed_collums_TimeStamp(data)
    # remove initial(first point) point from the columns TimeStamp , geometry,
    Timestamp_list  = []
    lines = []
    for n in range(len(data)):
        timestamp = data.at[n, 'TimeStamp']
        Timestamp_list.append(timestamp[1:])
        line = data.at[n,'geometry']
        new_line = sp.geometry.LineString(line.coords[1:])
        lines.append(new_line)
    data['TimeStamp'] = Timestamp_list
    data['geometry'] = lines
    data.to_parquet( outout_location)
    return data


if __name__ == '__main__':
    ds = combine_all_drifter_files(settings.DATA_DIR / 'Drifters')
    csv_parquet(ds, outout_location= settings.DRIFTER_DATA)
    csv_parquet(ds, outout_location= settings.DRIFTER_GEOFENCED_DATA_UNMAPPED , Geofenced_box= True)
