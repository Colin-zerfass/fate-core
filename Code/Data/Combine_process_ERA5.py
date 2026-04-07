"""
script to rename ERA5 vars into standard name for the project 

also can combine with previous ERA5 data to 

USE: cd Code\Data
python Combine_process_ERA5.py

Change file names 

"""


import xarray as xr

new_ds  = 'ca594838eb9afac1a235be5b42da7e33.nc'

output_name  = 'ERA5_10m_winds_2021_2026.nc'

combine = False ## controls if you want to combine with old dataset 
old_ds = 'ERA5_10m_winds.nc' # dataset to combine the new download with
rename_old_vars = False #rename if 

ds = xr.load_dataset(new_ds)
dsr = ds.rename({'valid_time': 'time'})
dsr = dsr.drop_vars(['expver', 'number'])
dsr = dsr.rename_vars({'u10':'uo', 'v10':'vo'})

if combine: 
    ds_old = xr.load_dataset(old_ds)
    if rename_old_vars:
        ds_oldr = ds_old.rename({'lat': 'latitude', 'lon': 'longitude'})
        ds_oldr = ds_oldr.drop_vars(['valid_time', 'number', 'step', 'surface'])
    
    merged = xr.merge([ds_oldr, dsr], join= 'outer', compat= 'override')
    merged = merged.sortby('time')
    merged = merged.drop_duplicates('time')
    dsr = merged

dsr.to_netcdf(output_name)




