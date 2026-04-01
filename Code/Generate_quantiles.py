import numpy as np 
import pandas as pd 
import geopandas as gpd
import functions.funcs as funcs 
import statsmodels.formula.api as smf 
import xarray as xr

"""Genrates Regression for forecast errors. Error(qauntile, leadtime, initial_speed_dif, latitude)


"""
## True dFAD data 
ds = gpd.read_parquet(r"Data\MappedOSCAR_SAT_MI_Cleanedspeeds.parquet") 

fc = pd.read_csv("Parcels/saved_output/Autocorrelation_intial_speed_dif_OSCAR_CMEMS_wind_2022_2025.csv")





fc["Time"] = pd.to_datetime(fc["Time"])
fc['error_km'] = funcs.haversine_df(fc, "lat_true", "lon_true", "lat_forcast", "lon_forcast")

def add_starttime(fc):
    fc["starttime"] = (fc["Time"] - pd.to_timedelta(fc["leadtime"], unit= "hours")).dt.round("min")
    return fc


## Unpacking True dFAD data into one list
longlist = pd.DataFrame({})
longlist["Time"] = funcs.Column_to_List(ds, "TimeStamp", idlist = False)
longlist["lats"], longlist["lons"] = funcs.list_of_latlon(ds, False)
longlist["x_speed"] = funcs.Column_to_List(ds, "x_speed", idlist = False)
longlist["y_speed"] = funcs.Column_to_List(ds, "y_speed", idlist = False)
longlist["v_mapped"], longlist["BuoyID"]  =funcs.Column_to_List(ds, "mapped_v", idlist = True)
longlist["v_mapped_OSCAR"], longlist["BuoyID"]  =funcs.Column_to_List(ds, "mapped_v_oscar", idlist = True)
longlist["u_mapped"] = funcs.Column_to_List(ds, "mapped_u", idlist = False)
longlist["u_mapped_OSCAR"] = funcs.Column_to_List(ds, "mapped_u_oscar", idlist = False)
longlist.Time = pd.to_datetime(longlist.Time)

def merge_forecast_true(fc, longlist):
    """Merges all True dFAD data into the format of a forecast output"""
    merged = pd.merge_asof(
    fc.sort_values('Time'),
    longlist.sort_values('Time'),
    on='Time',
    by='BuoyID',
    tolerance=pd.Timedelta(minutes=1),
    direction='nearest'
    )
    return merged

def calc_intial_speed_dif(merged, sufix= None):
    """ Calcuate the difference in speed between the true dFADs speeds and the models direction"""
    initial_speed_dif_mag = "initial_speed_dif_mag" 
    speed_dif_mag = "speed_dif_mag"
    u = "u_mapped"
    v = "v_mapped"
    if sufix is not None:
        ### if Sufix is oscar than it compairs with OSCAR 
        initial_speed_dif_mag +=sufix
        speed_dif_mag +=sufix
        u += sufix
        v+= sufix 
    merged[speed_dif_mag] = np.sqrt((merged.x_speed - merged[u])**2 + (merged.y_speed - merged[v])**2)
    merged[initial_speed_dif_mag] = merged.groupby(["BuoyID", "starttime"],  observed=False)[speed_dif_mag].transform("first")
    return merged

def calc_iniial_lat(merged):
    """Stores the intial latitudes onto all later leadtimes of that forecast"""
    merged = merged.sort_values("leadtime")
    merged["initial_lat"] = merged.groupby(["BuoyID", 'starttime'])["lat_true"].transform("first")
    return merged

merged = merge_forecast_true(fc, longlist)
merged = add_starttime(merged)
merged["speed"] = np.sqrt(merged.x_speed**2 + merged.y_speed**2)
merged = calc_intial_speed_dif(merged)
merged = calc_iniial_lat(merged)

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
    print(qrange)
    output = pd.DataFrame(columns = ["q" , "Intercept", 'initial_speed_dif_mag', "initial_lat"])
    for q in qrange:
        model = quantile_regreession_oneleatime_and_q(data,q)
        model = model.params.to_frame().T
        model["q"] = q
        output = pd.concat([output, model])
    return output

def quantile_regression(data, qstep = 0.1, timestep =4):
    timerange = np.arange(0,24*7,timestep )
    data['lead_bin'] = pd.cut(data.leadtime, timerange)
    # get the ordered bin intervals from the categorical produced by pd.cut when possible
    bins = timerange[1:]
    binlist = data.lead_bin.unique().sort_values().dropna()
    # match the qrange used in quantile_regression_one_leadtime (skip 0)
    qrange = np.arange(0,1,qstep)[1:]

    # use integer leadtime indices for the dataset dimension and store the actual
    # interval labels as a separate coordinate named 'lead_bin'
    op = xr.Dataset(
        {
            "Intercept": (["leadtime", "q"], np.full((len(bins), len(qrange)), np.nan)),
            "initial_speed_dif_mag": (["leadtime", "q"], np.full((len(bins), len(qrange)), np.nan)),
            "initial_lat": (["leadtime", "q"], np.full((len(bins), len(qrange)), np.nan)),
        },
        coords={"leadtime": np.arange(len(bins)), "q": qrange}
    )

    for i, bin in enumerate(binlist):
        print(bin)
        lt_bin = data.query('lead_bin == @bin')
        output = quantile_regression_one_leadtime(lt_bin, qstep=qstep)
        for _, row in output.iterrows():
            q_idx = np.where(qrange == row["q"])[0][0]
            op["Intercept"].values[i, q_idx] = row["Intercept"]
            op["initial_speed_dif_mag"].values[i, q_idx] = row["initial_speed_dif_mag"]
            op["initial_lat"].values[i, q_idx] = row["initial_lat"]

    return op

output = quantile_regression(merged)

output.to_netcdf(r"Data\regression_quantiles_leadtimes_2022_2024.nc")





