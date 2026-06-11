"""
script to rename ERA5 vars into standard name for the project 

also can combine with previous ERA5 data to 

USE: cd Code\Data
python Combine_process_ERA5.py

Change file names 

"""
import xarray as xr
import functions.settings as settings

combine_multiple_files = True
output_name  = 'ERA5_10m_winds_2021_2026v2.nc'
add_stokes_drift_variable = True

combine = True ## controls if you want to combine with old dataset 
old_ds = settings.ERA5_FILE # dataset to combine the new download with
rename_old_vars = False #rename 

if not combine_multiple_files: 
    new_ds  = '5387bce75d0b3240a113f74832571b48.nc'
    ds = xr.load_dataset(new_ds)
else: 
    import pathlib as path 
    download_data = settings.CORE_DIR / 'data_download'
    def Concatinate_multiple_files(files):
        ds_list = []
        for file in download_data.glob("*.nc"):
            print(file)
            ds = xr.load_dataset(file)
            ds_list.append(ds)
        
        return xr.concat(ds_list, dim = 'valid_time')

    ds = Concatinate_multiple_files(download_data)

dsr = ds.rename({'valid_time': 'time'})
dsr = dsr.drop_vars(['expver', 'number'])
if not add_stokes_drift_variable: 
    dsr = dsr.rename_vars({'u10':'uo', 'v10':'vo'})

dsr = dsr.sortby('latitude')
if combine: 
    ds_old = xr.load_dataset(old_ds)
    if rename_old_vars:
        ds_oldr = ds_old.rename({'lat': 'latitude', 'lon': 'longitude'})
        ds_old = ds_oldr.drop_vars(['valid_time', 'number', 'step', 'surface'])
    
    if not add_stokes_drift_variable:
        merged = xr.concat([ds_old, dsr], dim= 'time')
        merged = merged.sortby('time')
        merged = merged.drop_duplicates('time')
        dsr = merged
    if add_stokes_drift_variable:
        merged = xr.merge([ds_old, dsr], join='inner')
        dsr = merged
dsr.to_netcdf(output_name)
print(f'saved new file to {output_name}')
