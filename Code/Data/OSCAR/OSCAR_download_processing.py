
""" 
Process to reduce the size of OSCAR currents and compile into a zar file. 

USE: in the process argumrnt of podaac-data-downloader

'
podaac-data-downloader  -c OSCAR_L4_OC_INTERIM_V2.0 
                        -d Code\Data\OSCAR 
                        -sd 2025-07-01T00:00:00Z 
                        -ed 2026-01-01T00:00:00Z 
                        -b="-164,4.25,-160.5,8" 
                        --process ".venv\Scripts\python.exe Code\Data\OSCAR\OSCAR_download_processing.py"
'

podaac-data-downloader  -c OSCAR_L4_OC_INTERIM_V2.0 -d Code\Data\OSCAR -sd 2025-07-01T00:00:00Z -ed 2026-01-01T00:00:00Z -b="-164,4.25,-160.5,8" --process ".venv\Scripts\python.exe Code\Data\OSCAR\OSCAR_download_processing.py

"""

import sys 
import os
import xarray as xr

fname = sys.argv[1]

zarr_path = r"Code\Data\OSCAR\combined_oscar.zarr"


lat_min = 4.25
lat_max = 8
lon_min = -164+360
lon_max = -160.5 +360

tmp_fname = fname.replace(".nc", "_tmp.nc")
#ds_small = ds.sel(lat = slice(lat_min,lat_max), lon = slice(lon_min, lon_max))

ds = xr.open_dataset(fname)
ds_small = ds.sel(
    lon=slice(lon_min, lon_max),
    lat=slice(lat_min, lat_max)
).load()

if os.path.exists(zarr_path):
    ds_small.to_zarr(
        zarr_path,
        mode="a",
        append_dim="time"
    )
else:
    ds_small.to_zarr(
        zarr_path,
        mode="w"
    )

ds.close()
    
ds_small.to_netcdf(tmp_fname)

os.replace(tmp_fname, fname)