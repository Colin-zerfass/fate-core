import pandas as pd 
import numpy as np 

def add_starttime(fc):
    fc['Time'] = pd.to_datetime(fc['Time'])
    fc['leadtime_dt'] = pd.to_timedelta(fc['leadtime'], unit= 'hours')
    fc["starttime"] = (fc["Time"] - fc["leadtime_dt"]).dt.round("min")
    return fc

def haversine_df(df, lat1='lat1', lon1='lon1', lat2='lat2', lon2='lon2', radius=6371):
    """
    Compute the Haversine distance between two points for every row of a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing latitude/longitude columns
    lat1, lon1 : str
        Names of the first point's latitude/longitude columns
    lat2, lon2 : str
        Names of the second point's latitude/longitude columns
    radius : float
        Earth radius in meters (default is 6371000 m)

    Returns
    -------
    pandas.Series
        Distance (meters) for each row.
    """

    # Convert degrees → radians
    lat1_r = np.radians(df[lat1].values)
    lon1_r = np.radians(df[lon1].values)
    lat2_r = np.radians(df[lat2].values)
    lon2_r = np.radians(df[lon2].values)

    # Differences
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return radius * c

def calc_initial_lat_lon(ds):
    ds_sorted = ds.sort_values(['leadtime'])
    ds_sorted['lati'] = ds_sorted.groupby(['BuoyID', 'starttime'])['lat_true'].transform('first')
    ds_sorted['loni'] = ds_sorted.groupby(['BuoyID', 'starttime'])['lon_true'].transform('first')
    return ds_sorted

def calc_skillscore(dslist):
    singleinput = not isinstance(dslist, (list,tuple))
    if singleinput: 
        dslist = [dslist]
    for i, dsi in enumerate(dslist):  
        dsi = add_starttime(dsi)
        dsi = calc_initial_lat_lon(dsi)
        dsi['displacement'] = haversine_df(dsi, 'lati', 'loni', 'lat_forcast', 'lon_forcast')
        dsi['skillscore'] = 1 - (dsi.error_km/dsi.displacement)
        dsi['skillscore'] = dsi.skillscore.clip(lower = 0)
        dslist[i] = dsi
    return dslist[0] if singleinput else dslist

def merge_forecast_true(fc, longlist):
    """Combines Forecast data with the true dFAD data merging them into one data frame. """
    merged = pd.merge_asof(
    fc.sort_values('Time'),
    longlist.sort_values('Time'),
    on='Time',
    by='BuoyID',
    tolerance=pd.Timedelta(minutes=1),
    direction='nearest'
    )
    return merged

def calc_projection_initial_angle(merged, sufix = None):
    """Must be used after merge_forecast_true"""
    projection = "projection"
    angle = "angle"
    u = "u_mapped"
    v = "v_mapped"
    if sufix is not None:
        projection +=sufix
        angle +=sufix
        u += sufix
        v+= sufix 
    merged[projection] = (merged.y_speed * merged[v])+ (merged.x_speed * merged[u])
    merged[projection]  = merged[projection]/(merged.x_speed**2 +merged.y_speed**2 )**(1/2)
    merged[angle] = merged[projection]/(merged[u]**2 +merged[v]**2 )**(1/2)
    merged[angle] = np.arccos(merged[angle])*180/np.pi
    merged = merged.sort_values("leadtime")
    merged["initial_"+angle] =  merged.groupby(["BuoyID", "starttime"],  observed=False)[angle].transform("first")
    merged["initial_"+projection] = merged.groupby(["BuoyID", "starttime"],  observed=False)[projection].transform("first")
    return merged

def calc_intial_speed_dif(merged, sufix= None):
    """Must be used after merge_forecast_true"""
    initial_speed_dif_mag = "initial_speed_dif_mag" 
    speed_dif_mag = "speed_dif_mag"
    u = "u_mapped"
    v = "v_mapped"
    if sufix is not None:
        initial_speed_dif_mag +=sufix
        speed_dif_mag +=sufix
        u += sufix
        v+= sufix 
    merged[speed_dif_mag] = np.sqrt((merged.x_speed - merged[u])**2 + (merged.y_speed - merged[v])**2)
    merged[initial_speed_dif_mag] = merged.groupby(["BuoyID", "starttime"],  observed=False)[speed_dif_mag].transform("first")
    return merged

def calc_iniial_lat(merged):
    """Must be used after merge_forecast_true"""
    merged = merged.sort_values("leadtime")
    merged["initial_lat"] = merged.groupby(["BuoyID", 'starttime'])["lat_true"].transform("first")
    return merged

def inital_current_var(merged,sufix = None):
    """Must be used after merge_forecast_true"""
    import xarray as xr
    cmems = xr.open_dataset(r"Data\cmems.nc")
    varu = cmems.sel(latitude = slice(4.5, 7.5), depth = 15.81007).uo.var(dim = ["latitude", "longitude"])
    varv = cmems.sel(latitude = slice(4.5, 7.5), depth = 15.81007).vo.var(dim = ["latitude", "longitude"])
    varts = pd.DataFrame({"startday": varu.time.values, "varu": varu.values, "varv": varv.values})
    varts["var"] = varts.varu + varts.varv
    varts['startday'] = varts.startday.dt.date
    #merged["startday"] = merged.starttime.dt.date
    merged["startday"] = merged.groupby(['BuoyID', 'starttime'], observed= False)['starttime'].transform('first')
    merged['startday'] = merged['startday'].dt.date
    mergedvar = pd.merge(merged, varts, how = "left",on =  "startday")
    return mergedvar
