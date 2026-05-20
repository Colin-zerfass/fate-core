import pandas as pd
import geopandas as gpd
import shapely as sp
import numpy as np
from shapely.geometry import LineString, Polygon, Point
import functions.funcs as funcs




def split_trajectory_in_domain(line: LineString, domain: Polygon):
    """
    Splits a LineString trajectory into sub-trajectories (LineStrings)
    that lie strictly within a polygon domain.

    Rules:
    - Only keeps original vertices (no new boundary points added).
    - Only keeps sub-trajectories with >= 2 inside vertices.
    - Mask marks original vertices that belong to a valid inside sub-trajectory.

    Parameters
    ----------
    line : shapely.LineString
        Input trajectory.
    domain : shapely.Polygon
        Polygon domain.

    Returns
    -------
    sub_trajectories : list of shapely.LineString
        List of sub-trajectories inside the domain.
    mask : np.ndarray (bool)
        Boolean mask for each vertex of the original line.
    """
    from shapely.geometry import LineString, Polygon, Point
    coords = np.array(line.coords)
    inside = np.array([domain.contains(Point(p)) for p in coords])

    sub_trajectories = []
    mask = np.zeros_like(inside, dtype=bool)

    current_segment = []
    current_indices = []

    for i, (pt, flag) in enumerate(zip(coords, inside)):
        if flag:
            current_segment.append(pt)
            current_indices.append(i)
        else:
            # if segment ended, check length
            if len(current_segment) >= 2:
                sub_trajectories.append(LineString(current_segment))
                mask[current_indices] = True
            # reset
            current_segment = []
            current_indices = []

    # final check at end
    if len(current_segment) >= 2:
        sub_trajectories.append(LineString(current_segment))
        mask[current_indices] = True

    return sub_trajectories, mask

def remove_outof_domain(data:gpd.GeoDataFrame):
    boundry = sp.Polygon([(-163.75,4.5), (-163.75,7.8), (-160 +2/3, 7.8),(-160 +2/3, 4.5)])
    Values = []
    for n in range(len(data)):
        line = data["geometry"][n]
        lines , mask = split_trajectory_in_domain(line, domain = boundry)
        if len(lines) <1:
            Values.append(n)
    print(Values)
    data = data.drop(Values, axis = 0)
    data = data.reset_index()
    return data

def Add_x_y_speed_collums_TimeStamp(data): 
    """Converts distances in x, y, and xy to speeds in x,y,xy in m/s, using the sampleing frequency"""
    convert = 1000 ## km/hr to m/s 

    xspeeds = []
    yspeeds = []
    xyspeeds = []
    for i in data.index:
        xdistance = np.array(data.at[ i, "x_km"])
        xspeed = xdistance/data.at[i,"Delta_Timestamps"].astype("float64")*convert
        xspeeds.append(xspeed)

        ydistance = np.array(data.at[i, "y_km"])
        yspeed = ydistance/data.at[i,"Delta_Timestamps"].astype("float64")*convert
        yspeeds.append(yspeed)

        xydistance = np.array(data.at[i,"xy_km"])
        xyspeed = xydistance/data.at[i,"Delta_Timestamps"].astype("float64")*convert
        xyspeeds.append(xyspeed)

    data["x_speed"] = xspeeds
    data["y_speed"] = yspeeds
    data["xy_speed"] = xyspeeds
    return data

def add_delta_time_collums(data:gpd.GeoDataFrame):
    """Adds a collum that has a list of the change in times, Removes Time stamps that are Nan"""
    Times = []
    for i in range(len(data)):
        timelist = data.at[i, "TimeStamp"]
        timelist = pd.to_datetime(timelist, format = r"%Y-%m-%d %H:%M:%S")
        deltatime = np.diff(timelist)/10**9 ## converts from nano seconds
        Times.append(deltatime)
    data["Delta_Timestamps"] = Times
    return data

def remove_no_TimeStamp(data):
    data, xdistance, ydistance = funcs.add_distance_collumns(data)
    data = data.dropna(subset=["TimeStamp"])
    data = data.reset_index()
    return data

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

