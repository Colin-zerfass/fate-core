"""
Script to combine newly downloaded cmems data with previously downloaded cmems data. 
appending new data at the end of cmems data. 

"""

import xarray as xr 

new_file = 'cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m_uo-vo_163.83W-160.42W_4.42N-7.83N_0.49-29.44m_2025-01-01-2026-01-01.nc'

old_file = 'cmems.nc'

output_name = 'cmems_2022_2026.nc'

ds = xr.load_dataset(new_file)

ds_old = xr.load_dataset(old_file)

ds = ds.sel(
    latitude=slice(ds_old.latitude.min(), ds_old.latitude.max()),
    longitude=slice(ds_old.longitude.min(), ds_old.longitude.max()),
    depth=slice(ds_old.depth.min(), ds_old.depth.max())
)

merged = xr.merge([ds_old, ds], join='outer', compat='override')
merged = merged.sortby('time')
merged = merged.drop_duplicates('time')

print(merged.latitude.values.max(), merged.latitude.values.min())
print(merged.longitude.values.max(), merged.longitude.values.min())
print(merged.time.values.max(), merged.time.values.min())

merged.to_netcdf(output_name)