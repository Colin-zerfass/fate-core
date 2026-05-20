"""
script to rename ERA5 vars into standard name for the project 

also can combine with previous ERA5 data to 

USE: cd Code\Data
python Combine_process_ERA5.py

Change file names 

"""


import xarray as xr

new_ds  = '5387bce75d0b3240a113f74832571b48.nc'

output_name  = 'ERA5_10m_winds_2021_2026v2.nc'

combine = True ## controls if you want to combine with old dataset 
old_ds = 'ERA5_10m_winds_2021_2026.nc' # dataset to combine the new download with
rename_old_vars = False #rename if 

ds = xr.load_dataset(new_ds)
dsr = ds.rename({'valid_time': 'time'})
dsr = dsr.drop_vars(['expver', 'number'])
dsr = dsr.rename_vars({'u10':'uo', 'v10':'vo'})
dsr = dsr.sortby('latitude')

if combine: 
    ds_old = xr.load_dataset(old_ds)
    if rename_old_vars:
        ds_oldr = ds_old.rename({'lat': 'latitude', 'lon': 'longitude'})
        ds_old = ds_oldr.drop_vars(['valid_time', 'number', 'step', 'surface'])
    
    merged = xr.concat([ds_old, dsr], dim= 'time')
    merged = merged.sortby('time')
    merged = merged.drop_duplicates('time')
    dsr = merged

dsr.to_netcdf(output_name)




