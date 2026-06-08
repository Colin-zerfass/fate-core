"""
Goal: Combine drifters into the same dataformat as the main dFAD ds. 
parquet where each trajecotry is its own row. Locations are stored as polylines
1) Columns needed. BuoyName, MinOfTimes, MaxOfTimes, TimeStamp, geometry, 
x_deg, y_deg, x_km, y_km, xy_km, x_speed, y_speed, xy_speed
"""

import pandas as pd
import glob 
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


def csv_parquet(data:pd.DataFrame, add_min_max_times = True) -> gpd.GeoDataFrame:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    if add_min_max_times: 
        data["MinOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("min")
        data["MaxOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("max")

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

    data.to_parquet(settings.DATA_DIR/ 'Drifter_cleaned_2026_06.parquet')
    return data


if __name__ == '__main__':
    ds = combine_all_drifter_files(settings.DATA_DIR / 'Drifters')
    ds = csv_parquet(ds)
