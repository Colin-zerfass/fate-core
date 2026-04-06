"""File to combine the indivagual days of the OSCAR data into one netCDF file """ 
import xarray as xr 
import numpy as np
import glob 
import os 
import pandas as pd

output_file = r"..\OSCAR_combined_2021_2025.nc"

files = sorted(glob.glob("oscar_currents_interim_*.nc"))

ds = xr.open_mfdataset(files, combine= "by_coords")

time_index = pd.to_datetime([t.isoformat() for t in ds.time.values])

ds["time"] = time_index

print(output_file)

ds.to_netcdf(output_file)
