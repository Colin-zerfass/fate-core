import pandas as pd 
import numpy as np 
import geopandas as gpd 
from pykrige.ok import OrdinaryKriging
import xarray as xr
from functions.funcs import *

ds = gpd.read_parquet(r"Code\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")

def querry_date(data:gpd.GeoDataFrame, date)-> gpd.GeoDataFrame:
    """Returns only dFADs that are active during this time period"""
    data = data.query("MinOfDate <= @date")
    data = data.query("MaxOfDate >= @date")
    return data

ds = ds.reset_index(drop=True )

## get range of start and stop dates
start_date = ds.MinOfDate.min().date()
end_date = ds.MaxOfDate.max().date()
timedelta = pd.Timedelta(days= 1) ##How often should this make a forcast 
date_range = pd.date_range(start_date, end_date, freq=timedelta)

#making the dataset to save the time series. 
data = xr.open_dataset(r"Code\cmems.nc") ##using same cordinates as cmems. 
lats = data.latitude
lons = data.longitude
uos = []
vos = []

for target_date in date_range:

    ds_active = querry_date(ds, date = target_date) ## All of the active dFADs at this time 
    ds_active = ds_active.reset_index()

    columns = ["TimeStamp", "x_speed", "y_speed"]
    ds_locations = pd.DataFrame()
    for label in columns: 
        longlist = Column_to_List(ds_active, label) 
        ds_locations[label] = longlist
    lat, lon  = list_of_latlon(ds_active, droplast= False)
    ds_locations["lat"] = lat
    ds_locations["lon"] =lon
    ds_locations.TimeStamp = pd.to_datetime(ds_locations.TimeStamp)

    ##Filter Timestep by certain threshhold to get locations of FADS within closes  
    ## UPDATE:This might be better to interp these onto the specific time. 

    time_upper  = target_date 
    time_lower = target_date - timedelta
    ds_locations = ds_locations.query(f"TimeStamp > @time_lower")
    ds_locations = ds_locations.query(f"TimeStamp < @time_upper")
    ##Removes repeated coords when the dFAD hardly moves 
    ds_locations = ds_locations.drop_duplicates(subset=["lat"], keep="first")
    ds_locations = ds_locations.drop_duplicates(subset=["lon"], keep="first")

    print(f"{target_date.date()}:Amount of sampled dFAD within {timedelta} hrs : {len(ds_locations)}")
    if len(ds_locations) > 5: ##This is the method od Krigging, can mess with this/change this to see how this works
        x = OrdinaryKriging(
                ds_locations.lon, ds_locations.lat, ds_locations.x_speed,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False
            )

        y = OrdinaryKriging(
                ds_locations.lon, ds_locations.lat, ds_locations.y_speed,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False
            )
        ##Need to set lat lons Size 
        z_predx, ss_x = x.execute('grid', lons, lats)
        z_predy, ss_y = y.execute('grid', lons, lats)
    if len(ds_locations) <= 1: 
        z_predx = np.zeros([len(lats), len(lons)])
        z_predy = np.zeros([len(lats), len(lons)])
    uos.append(z_predx)
    vos.append(z_predy)

uo = np.stack(uos, axis = 0)
vo = np.stack(vos,axis = 0)
uo_da = xr.DataArray(
    uo,
    coords={"time": date_range, "lat": lats.data, "lon": lons.data},
    dims=("time", "lat", "lon")
)

vo_da = xr.DataArray(
    vo,
    coords={"time": date_range, "lat": lats.data, "lon": lons.data},
    dims=("time", "lat", "lon")
)

ds_krigg = xr.Dataset({"uo": uo_da, "vo": vo_da})

ds_krigg.to_netcdf("Code\Data\kirgging_field.nc")