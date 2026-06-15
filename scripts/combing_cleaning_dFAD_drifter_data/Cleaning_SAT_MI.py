"""From a .csv containing dFAD data, puts into standard form by producing a .parquet file, 
This also cleans all of the data removing high and low speed points making all lists within the same row have the same speed. 
Future work could seperate the dFAD trajectories that leave the area for over 1 day(or some amount) into a new tajectorty"""


import pandas as pd
import geopandas as gpd
import shapely as sp
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import functions.Cleaning_functions as cleaning
import functions.funcs as funcs 
import functions.settings as settings 


def Cleaning_SAT_MI():
    data= pd.read_csv(settings.DATA_DIR / "SAT_MI_FAD.csv", low_memory=False )

    ##First combine new data into linestrings and standard Format
    Latitude_list = data.groupby("BuoyName")["Latitude"].apply(list)
    Longitude_list = data.groupby("BuoyName")["Longitude"].apply(list)
    times_list = data.groupby("BuoyName")["Timestamp"].apply(list)
    date_enter = data.groupby("BuoyName")["MinOfTimes"].apply(np.minimum.reduce)
    date_exit = data.groupby("BuoyName")["MaxOfTimes"].apply(np.minimum.reduce)
    BuoyNames = data["BuoyName"].unique()

    print(len(Latitude_list))
    mask = Latitude_list.apply(len) != 1
    Latitude_list = Latitude_list[mask]
    Longitude_list = Longitude_list[mask]
    times_list = times_list[mask]
    date_enter = date_enter[mask]
    date_exit = date_exit[mask]

    BuoyNames_filtered = Latitude_list.index.to_numpy()
    print(len(Latitude_list))

    Latitude_list
    lines = []
    for n in range(len(Latitude_list)):
        line = sp.linestrings(Longitude_list.iloc[n],Latitude_list.iloc[n])
        lines.append(line)


    print(len(BuoyNames), len(lines), len(date_enter), len(date_exit))
    newdata = gpd.GeoDataFrame({"BuoyName":BuoyNames_filtered, "MinOfDate":date_enter, "MaxOfDate": date_exit, "TimeStamp": times_list, "geometry": lines}, index= None)
    newdata = newdata.reset_index(drop = True)

    data = newdata
    data = cleaning.remove_outof_domain(data)

    data, delx_long, dely_long= funcs.add_distance_collumns(data)
    data = cleaning.remove_no_TimeStamp(data)
    data = funcs.add_delta_time_collums(data)
    data = cleaning.Add_x_y_speed_collums_TimeStamp(data)
    rowtestingg = 300
    print(f"checking the size of array to see if they are the same row {rowtestingg}")
    print(f"length of x_km: {len(data.x_km[rowtestingg])}")
    x,y = data.geometry[rowtestingg].xy
    print(f"Number of coords: {len(x)}")
    print(f"For Buoy {data.BuoyName[rowtestingg]}")

    
    ## Checking how many points that need to be removed 
    speed = np.array(funcs.Column_to_List(data,"xy_speed"))
    speed_fast = speed[speed>2]
    print(f"high speed: {len(speed_fast)/len(speed)*100}")

    speed_slow = speed[speed<0.0001]
    print(f"Slow Speeds: {len(speed_slow)/len(speed)*100:.3}%")

    deltatimes = np.array(funcs.Column_to_List(data,"Delta_Timestamps"))


    dataclean = cleaning.Remove_speeds_high_low(data)
    print(dataclean["points_removed"].sum())
    dataclean = cleaning.Remove_zero_timedeltas(dataclean)
    print(dataclean["points_removed2"].sum())
    dataclean = cleaning.Combine_masks(dataclean)
    dataclean = dataclean.reset_index(drop = True)

    print("Created mask for Speed and Time errors... Aplying mask")

    ### Removing The bad points in Geometry 
    dataclean["new_geometry"] = dataclean.apply(cleaning.Filter_geometry_obj, axis = 1)
    dataclean["geometry"] = dataclean["new_geometry"]
    dataclean = dataclean.drop(columns = ["new_geometry"])
    valid_mask = ~dataclean.geometry.isna() & ~dataclean.geometry.is_empty
    dataclean = dataclean[valid_mask]

    print("Applied mask")
    ### Removing Unneeded columns 

    columnlist = ['x_deg', 'y_deg', 'x_km', 'y_km', 'xy_km', 'TimeStamp', "x_speed","y_speed", "xy_speed","Delta_Timestamps" ]

    for names in columnlist: 
        print("Filtered Column :{names}")
        dataclean[f"new_{names}"] = dataclean.apply(cleaning.Filter_Rows, axis =1, column = names)
        dataclean[f"{names}"] = dataclean[f"new_{names}"] 
        dataclean = dataclean.drop(columns = [f"new_{names}"])

    dataclean = dataclean.reset_index(drop = True)

    print("---------------------")
    print("Data has been cleaned ")
    print("---------------------")

    ##Checking to see if it is cleaned
    Delta_Timestamps = funcs.Column_to_List(dataclean,"Delta_Timestamps")
    below1s = np.sum(Delta_Timestamps <= np.timedelta64(0, "ns"))
    print(f"Amount of zero delta times: {below1s}")
    speeds  = np.array(funcs.Column_to_List(dataclean,"xy_speed"))
    print(f"Number of nan speeds :{np.sum(np.isnan(speeds))}")

    if True: ## Checking size of the arrays
        print(f"checking the size of array to see if they are the same row {rowtestingg}")
        print(f"length of x_km: {len(dataclean.x_km[rowtestingg])}")
        x,y = dataclean.geometry[rowtestingg].xy
        print(f"Number of coords: {len(x)}")
        print(f"Number of points removed in row {rowtestingg}: {dataclean.points_removed[rowtestingg] + dataclean.points_removed[rowtestingg]} ")
        print(f"For Buoy {dataclean.BuoyName[rowtestingg]}")

    dataclean.to_parquet(settings.dFAD_DATA_UNMAPPED)
    print(len(dataclean))
    speeds = funcs.Column_to_List(dataclean,"xy_speed")
    print(f"Max speeds are : {max(speeds)}")

if __name__ == '__main__': 
    Cleaning_SAT_MI()
