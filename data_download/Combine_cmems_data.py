"""
Script to combine newly downloaded cmems data with previously downloaded cmems data. 
appending new data at the end of cmems data. 

"""
import xarray as xr 
import functions.settings as settings 

new_file = r'cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_163.83W-160.42W_4.42N-7.83N_109.73m_2021-06-01-2026-01-01.nc'
old_file = 'cmems_2021_2026v2.nc'
output_name = 'cmems_2021_2026v3.nc'

Merge = False ## for adding new depth or dims into the file 

ds = xr.load_dataset(new_file)
ds_old = xr.load_dataset(old_file)

# Sort coordinates ascending so slice(min, max) works regardless of storage order
ds     = ds.sortby(['latitude', 'longitude', 'depth'])
ds_old = ds_old.sortby(['latitude', 'longitude', 'depth'])

# Round coordinates to 4 decimal places to eliminate floating-point discrepancies
decimals = 4
ds = ds.assign_coords(
    latitude=ds.latitude.values.round(decimals),
    longitude=ds.longitude.values.round(decimals),
    depth=ds.depth.values.round(decimals),
)
ds_old = ds_old.assign_coords(
    latitude=ds_old.latitude.values.round(decimals),
    longitude=ds_old.longitude.values.round(decimals),
    depth=ds_old.depth.values.round(decimals),
)

if Merge: 
    ds = ds.sel(time = slice(ds_old.time.min(), ds_old.time.max()))
    merged = xr.concat([ds_old, ds], dim = 'depth')
    merged = merged.interpolate_na(dim='latitude').interpolate_na(dim='longitude') ## fills depth values that are nan 
else: 
# Concat along time then deduplicate (avoids merge silently dropping new data)
    merged = xr.concat([ds_old, ds], dim='time')
merged = merged.sortby('time')
merged = merged.drop_duplicates('time')

print(merged.latitude.values.max(), merged.latitude.values.min())
print(merged.longitude.values.max(), merged.longitude.values.min())
print(merged.time.values.max(), merged.time.values.min())
print(merged.depth.values.max(), merged.depth.values.min())
print(merged.uo.isnull().sum())
breakpoint()
merged.to_netcdf(output_name)