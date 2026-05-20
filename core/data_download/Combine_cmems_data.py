"""
Script to combine newly downloaded cmems data with previously downloaded cmems data. 
appending new data at the end of cmems data. 

"""

import xarray as xr 

new_file = 'cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_163.83W-160.42W_4.42N-7.83N_0.49-29.44m_2021-06-27-2021-07-01.nc'

old_file = 'cmems_2021_2026.nc'

output_name = 'cmems_2021_2026v2.nc'

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

# Concat along time then deduplicate (avoids merge silently dropping new data)
merged = xr.concat([ds_old, ds], dim='time')
merged = merged.sortby('time')
merged = merged.drop_duplicates('time')

print(merged.latitude.values.max(), merged.latitude.values.min())
print(merged.longitude.values.max(), merged.longitude.values.min())
print(merged.time.values.max(), merged.time.values.min())

merged.to_netcdf(output_name)