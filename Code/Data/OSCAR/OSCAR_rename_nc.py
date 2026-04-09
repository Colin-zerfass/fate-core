"""File to turn from zarr into stamdard .nc file used in this code base. 
Conists of fixing time indexs, 
renaming and changing order of lat and lon,

Also, Can append OSCAR with a previous download of oscar. 
 """ 
import xarray as xr 
import numpy as np
import glob 
import os 
import pandas as pd

output_file = r"..\OSCAR_combined_2021_2026v2.nc"

zarr_file = r'C:\FATE\Code\Data\OSCAR\combined_oscar.zarr'

merge = True 
merge_with = r'..\OSCAR_combined_2021_2026.nc'

rename_old = False


ds = xr.open_zarr(zarr_file)
ds = ds.swap_dims({ 'latitude': 'lat', 'longitude': 'lon' }).rename({ 'lat': 'latitude', 'lon': 'longitude' })
ds['longitude'] = ds['longitude'] - 360


time_index = pd.to_datetime([t.isoformat() for t in ds.time.values])
ds["time"] = time_index

dst = ds.transpose('time', 'latitude', 'longitude')
dstr = dst.rename({ 'u': 'uo', 'v': 'vo' }).drop_vars(['ug', 'vg'])
dstr = dstr.drop_duplicates('time') ## removes duplicates of there are any in the zarr

if merge: 
    old = xr.open_dataset(merge_with)
    if rename_old: 
        old = old.swap_dims({ 'latitude': 'lat', 'longitude': 'lon' }).rename({ 'lat': 'latitude', 'lon': 'longitude' })
        oldt = old.transpose('time', 'latitude', 'longitude')
        old = oldt.rename({ 'u': 'uo', 'v': 'vo' }).drop_vars(['ug', 'vg'])

    dstr = xr.merge([old, dstr], join = 'outer')

dstr.to_netcdf(output_file)
