import pandas as pd
import geopandas as gpd
import shapely as sp
import numpy as np
from shapely.geometry import LineString, Polygon, Point


def haversine_dist_with_negetive(lat1, lon1, lat2, lon2):
    """ONLY to be used when calcuating Distance in X and Y indivigually 
    Returns distance between two points in Km 
    - xdirection and negetive y direction are allowed"""
    from math import radians, sin, cos, atan2, sqrt
    # Convert lat and lon to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
 
    dist_lon = lon2_rad-lon1_rad
    dist_lat = lat2_rad-lat1_rad
 
    a = sin(dist_lat/2)**2+cos(lat1_rad)*cos(lat2_rad)*sin(dist_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
 
    R = 6371.0
 
    distance = R*c
    if lat1 > lat2: ## manually checks if Values should be negetive 
        distance = -distance
    if lon1 >lon2:
        distance =-distance
    return distance

def haversine_dist(lat1, lon1, lat2, lon2):
    """Returns distance between two points in Km (always positive)"""
    from math import radians, sin, cos, atan2, sqrt
    # Convert lat and lon to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
 
    dist_lon = lon2_rad-lon1_rad
    dist_lat = lat2_rad-lat1_rad
 
    a = sin(dist_lat/2)**2+cos(lat1_rad)*cos(lat2_rad)*sin(dist_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
 
    R = 6371.0
 
    distance = R*c
    return distance

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

def add_distance_collumns(data = gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    """Adds distance dfad traveled in x and y directions, also provides a list of all points"""
    delx_list = []
    dely_list = []
    test_data = data
    delx_long = np.array([])
    dely_long = np.array([])

    lines = test_data["geometry"]
    length = lines.shape[0]

    delxs = []
    delys=[]
    x_distkms = []
    y_distkms = []
    xy_distkms = []
    # bigger loop 

    for n in range(0,length):
        singleline = lines[n]
        coords = list(lines[n].coords)

        x_coords = []
        y_coords = []
        for x, y in coords:
             x_coords.append(x)
             y_coords.append(y)

        x_distkm = []
        y_distkm = []
        xy_distkm = []
        

        for n in range(0,len(x_coords)-1): ##this has to be one less because using n+1
            m = int(n+1)
            xdist = haversine_dist_with_negetive(x_coords[n], y_coords[n], x_coords[m],y_coords[n])## gives just the x distance traveled
            ydist = haversine_dist_with_negetive(x_coords[n], y_coords[n], x_coords[n],y_coords[m]) ## gives just the y distance traveled
            xydist = haversine_dist(x_coords[n], y_coords[n], x_coords[m],y_coords[m]) ##gives distance in both xy
            x_distkm.append(xdist) 
            y_distkm.append(ydist)
            xy_distkm.append(xydist)

        delx = np.diff(x_coords)
        dely=np.diff(y_coords)
        x_distkms.append(x_distkm)
        y_distkms.append(y_distkm)
        xy_distkms.append(xy_distkm)

        delxs.append(delx)
        delys.append(dely)

        delx_long = np.append(delx_long, delx)
        dely_long = np.append(dely_long, dely)

    data["x_deg"] = delxs
    data["y_deg"] = delys
    data["x_km"] = x_distkms
    data["y_km"] = y_distkms
    data["xy_km"] = xy_distkms

    heading = np.atan(delx_long/dely_long)
    delx_long = np.cos(heading)
    dely_long = np.sin(heading)

    return data, delx_long, dely_long

def remove_no_TimeStamp(data):
    data, xdistance, ydistance = add_distance_collumns(data)
    data = data.dropna(subset=["TimeStamp"])
    data = data.reset_index()
    return data

def Column_to_List(data, column:str, idlist = False):
    """Returns a column as a long list of values"""
    ids = []
    long_list = []
    for i in range(len(data)):
        row = data.at[i, column]
        id = [data.at[i, "BuoyName"]]
        id = id*len(data.at[i, "TimeStamp"])
        long_list.extend(row)
        ids.extend(id)
    if idlist == True: 
        return long_list, ids 
    return long_list

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

