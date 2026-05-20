import pandas as pd 
import numpy as np 
import statsmodels.formula.api as smf 
import geopandas as gpd

def add_starttime(fc):
    fc = fc.copy()
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
 
def calculate_rmse(group, column = 'error_km'):
    """Calcuates RMSE, 
    USE:  ds.groupby('leadtime', observed = False).appy(calculate_rmse)"""
    rmse = np.sqrt((group[column]**2).mean())
    return rmse 

def interpolate_data(group, dt = 4):
    "intepolates the forecast data "

    lead_col = 'leadtime'
    bins = np.linspace(0, 8*24, 2*24 + 1)  # every 4 hours

    g = group.copy()
    g = g.set_index(lead_col).sort_index()

    # If there are duplicate leadtime values, remove them before interpolating
    if g.index.duplicated().any():
        g = g[~g.index.duplicated(keep='first')]
    
    maxleadtime = g.index.max()
    bins_shortened = bins[bins < maxleadtime]
    new_index = pd.Index(bins_shortened, name=lead_col)

    # include existing points so interpolation has anchors, then interpolate
    combined_index = new_index.union(g.index)
    g = g.reindex(combined_index).sort_index()
    cols = ['Time','lat_true', 'lon_true','lat_forcast', 'lon_forcast']
    g = g[cols].interpolate(method='linear', limit_direction='both')

    # keep only the rows at the bin locations
    out = g.reindex(new_index).reset_index()
    return out

def dtrue(group): 
    "calcuates the displacement of each timestep of the true data"
    group['dlat_true'] = group['lat_true'].diff()
    group['dlon_true'] = group['lon_true'].diff()
    return group


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



def quantile_regreession_oneleatime_and_q(data, q):
    """Data is at one specific leadtiem, q is what quantile (0-1)"""
    model = smf.quantreg("error_km ~ initial_speed_dif_mag + initial_lat", data)
    model = model.fit(q=q)
    return model

def quantile_regression_one_leadtime(data,qstep = 0.1):
    """ Calcuates all q for within that range. 
    Data has to already be filtered to one leadtime"""
    qrange = np.arange(0,1,qstep)
    qrange = qrange[1:]
    rows = []
    for q in qrange:
        model = quantile_regreession_oneleatime_and_q(data,q)
        model = model.params.to_frame().T
        model["q"] = q
        rows.append(model)
    output = pd.concat(rows, ignore_index=True)
    return output

def quantile_regression(data, qstep = 0.1, timestep =4):
    """calcs regression based on initial speed differance and latitude from forecasts
    Used 'Generate_qunatiles.py' and to make fig 5 in the paper
    """
    import xarray as xr


    timerange = np.arange(0,24*7,timestep )
    data['lead_bin'] = pd.cut(data.leadtime, timerange)
    # get the ordered bin intervals from the categorical produced by pd.cut when possible
    bins = timerange[1:]
    binlist = data.lead_bin.unique().sort_values().dropna()
    # match the qrange used in quantile_regression_one_leadtime (skip 0)
    qrange = np.arange(0,1,qstep)[1:]

    # store actual leadtime hours (upper edge of each bin) as the coordinate
    # so that outputs.sel(leadtime=72) correctly finds the 72-hour bin
    op = xr.Dataset(
        {
            "Intercept": (["leadtime", "q"], np.full((len(bins), len(qrange)), np.nan)),
            "initial_speed_dif_mag": (["leadtime", "q"], np.full((len(bins), len(qrange)), np.nan)),
            "initial_lat": (["leadtime", "q"], np.full((len(bins), len(qrange)), np.nan)),
        },
        coords={"leadtime": bins.astype(float), "q": qrange}
    )

    for bin in binlist:
        # use bin.right to find the correct position in the bins array,
        # so empty bins don't shift the index
        bin_idx = np.where(bins == bin.right)[0]
        if len(bin_idx) == 0:
            continue
        bin_idx = bin_idx[0]
        print(bin)
        lt_bin = data.query('lead_bin == @bin')
        output = quantile_regression_one_leadtime(lt_bin, qstep=qstep)
        for _, row in output.iterrows():
            q_idx = np.where(qrange == row["q"])[0][0]
            op["Intercept"].values[bin_idx, q_idx] = row["Intercept"]
            op["initial_speed_dif_mag"].values[bin_idx, q_idx] = row["initial_speed_dif_mag"]
            op["initial_lat"].values[bin_idx, q_idx] = row["initial_lat"]

    return op


def Projection_binning(merged: pd.DataFrame, label:str, binindex : int):
    bins = np.linspace(0,8*24,2*24+1)
    merged["lead_bins"] = pd.cut(merged["leadtime"], bins)
    binlist = merged["lead_bins"].unique().sort_values()
    a  =binlist[binindex] ## list of bin intervals 
    mergedhr = merged.groupby("lead_bins",  observed=False).get_group(a).copy()
    ##now group by speeds and take a mean.
    speedbins = np.linspace(mergedhr[label].min(),mergedhr[label].max(),20)
    mergedhr["projection_bin"] = pd.cut(mergedhr[label], speedbins)
    binned_errors = mergedhr.groupby("projection_bin",  observed=False)["error_km"].mean()
    return speedbins, binned_errors


def merged_dataframe_add_all_columns(forecast:pd.DataFrame, dFAD:gpd.GeoDataFrame):
    """Combine a forecast dataset with the dFADs using merge_fore_cast_true() 
    then adds the columns of starttime, initial_lat and inital_speed_dif """
    from functions.funcs import generate_longlist
    longlist = generate_longlist(dFAD, extra_columns= ['mapped_u', 'mapped_v'])
    longlist = longlist.rename(columns = {'mapped_u':'u_mapped', 'mapped_v': 'v_mapped'})
    merged  = merge_forecast_true(forecast, longlist)
    merged =  add_starttime(merged)
    merged =  calc_iniial_lat(merged)
    merged =  calc_intial_speed_dif(merged)
    merged = merged.sort_values(['BuoyID', 'starttime', 'Time']).reset_index(drop = True)
    return merged