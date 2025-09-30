# #### Script to clean dFAD data with the speeds and consecative times. 
# - ERORR: FINAL CLEANED DATA COULD HAVE NONE TYPE GEOMETRY LISTS, NEED TO FILTER THESE OUT
# - remove zero speeds
# - high speeds
# - remove dFADs with few data points or lots bad datapoints
# - from the removed points also have to remove the Lat, lon, distance, time collumn,

 
import pandas as pd 
import numpy as np 
import geopandas as gpd 
import matplotlib.pyplot as plt 
from functions.funcs import *
import sys 


if len(sys.argv) ==1:
    filename = sys.argv[1]
else:
    filename = "Data\Palmyra Data\SAT_MI_FAD_Missing_Times.parquet"

print(f"Opening File:{filename}") 
data = gpd.read_parquet(r"Data\Palmyra Data\SAT_MI_FAD_Missing_Times.parquet")

 
data, delx_long, dely_long= add_distance_collumns(data)
data = remove_no_TimeStamp(data)
data = add_delta_time_collums(data)
data = Add_x_y_speed_collums_TimeStamp(data)
 
## Checking how many points that need to be removed 
speed = np.array(Column_to_List(data,"xy_speed"))
speed_fast = speed[speed>2]
print(f"high speed: {len(speed_fast)/len(speed)*100}")

speed_slow = speed[speed<0.0001]
print(f"Slow Speeds: {len(speed_slow)/len(speed)*100:.3}%")

deltatimes = np.array(Column_to_List(data,"Delta_Timestamps"))

 
def Remove_speeds_high_low(data:gpd.GeoDataFrame):
    bad_points = []
    speeds = []
    Masks = [ ]
    for i in range(len(data)):
        speed = data.at[i, "xy_speed"]
        speed_high = speed>2
        speed_low = speed < 0.001
        masked = speed_high | speed_low
        masked =~ masked
        filtered_speed = speed[masked]
        bad_point = len(speed) - len(filtered_speed)
        
        speeds.append(filtered_speed)
        bad_points.append(bad_point)
        Masks.append(masked)
    
    dataclean = data.copy()
    #dataclean["xy_speed"] = speeds
    dataclean["points_removed"] = bad_points
    dataclean["Masked_array"] = Masks
    return dataclean 

def Remove_zero_timedeltas(data:gpd.GeoDataFrame):
    bad_points = []
    timedeltas = []
    masks = []
    for i in range(len(data)):
        dtimes = data.at[i,"Delta_Timestamps"]
        mask = np.array(dtimes) != np.timedelta64(0, "ns")
        filtered_delta = np.array(dtimes)[mask]
        bad_point = len(dtimes) - len(filtered_delta)
        timedeltas.append(filtered_delta)
        bad_points.append(bad_point)
        masks.append(mask)
    #data["Delta_Timestamps"] = timedeltas
    data["points_removed2"] = bad_points
    data["Masked_array2"] = masks

    return data
 
def Combine_masks(data):
    """
    Combine Masked_array from speed filter and Delta_Timestamps filter.
    Assumes both are boolean arrays of the same length per row.
    """
    combined_masks = []
    
    for i in range(len(data)):
        mask_speed = np.array(data.at[i, "Masked_array"])
        mask_delta = np.array(data.at[i, "Masked_array2"])
        
        # combine: keep points that pass both
        combined = mask_speed & mask_delta
        
        combined_masks.append(combined)
    
    dataclean = data.copy()
    dataclean["Masked_array_combined"] = combined_masks
    
    return dataclean

def Filter_geometry_obj(row):
    coords = np.asarray(row.geometry.coords)
    coords = coords[1:]
    filtered_coords = coords[row['Masked_array_combined']]
    if len(filtered_coords) > 1:
        return sp.geometry.LineString(filtered_coords)
    else:
    # Return an empty geometry if not enough points remain
        return None

def Filter_Rows(row,column):
    """Function implented to apply mask from bad point removal to to other columns
    if column size doesnt line up we remove the later point [1:]"""
    array =np.asarray(row[f"{column}"])
    if len(array) != len(row["Masked_array_combined"]):
        array = array[1:]
    filtered_data = array[row["Masked_array_combined"]]
    if len(filtered_data)>1:
        return filtered_data
    else:
        return None


if __name__ == "__main__":
    dataclean = Remove_speeds_high_low(data)
    print(dataclean["points_removed"].sum())
    dataclean = Remove_zero_timedeltas(dataclean)
    print(dataclean["points_removed2"].sum())
    dataclean = Combine_masks(dataclean)
    dataclean = dataclean.reset_index(drop = True)

    print("Created mask for Speed and Time errors... Aplying mask")
    
    ### Removing The bad points in Geometry 
    dataclean["new_geometry"] = dataclean.apply(Filter_geometry_obj, axis = 1)
    dataclean["geometry"] = dataclean["new_geometry"]
    dataclean = dataclean.drop(columns = ["new_geometry"])
    valid_mask = ~dataclean.geometry.isna() & ~dataclean.geometry.is_empty
    dataclean = dataclean[valid_mask]

    print("Applied mask")
    ### Removing Unneeded columns 

    columnlist = ['x_deg', 'y_deg', 'x_km', 'y_km', 'xy_km', 'TimeStamp', "x_speed","y_speed", "xy_speed","Delta_Timestamps" ]

    for names in columnlist: 
        print("Filtered Column :{names}")
        dataclean[f"new_{names}"] = dataclean.apply(Filter_Rows, axis =1, column = names)
        dataclean[f"{names}"] = dataclean[f"new_{names}"] 
        dataclean = dataclean.drop(columns = [f"new_{names}"])

    dataclean = dataclean.reset_index(drop = True)

    print("---------------------")
    print("Data has been cleaned ")
    print("---------------------")

    ##Checking to see if it is cleaned
    Delta_Timestamps = Column_to_List(dataclean,"Delta_Timestamps")
    below1s = np.sum(Delta_Timestamps <= np.timedelta64(0, "ns"))
    print(f"Amount of zero delta times: {below1s}")
    speeds  = np.array(Column_to_List(dataclean,"xy_speed"))
    print(f"Number of nan speeds :{np.sum(np.isnan(speeds))}")

    
    dataclean.to_parquet(r"Data\Palmyra Data\SAT_MI_FAD_cleanedspeeds.parquet")

    speeds = Column_to_List(dataclean,"xy_speed")
    print(f"Max speeds are : {max(speeds)}")



