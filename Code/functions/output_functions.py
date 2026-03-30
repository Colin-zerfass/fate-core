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