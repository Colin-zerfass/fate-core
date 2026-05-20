"""Python script to go from monthly climatology to interpolating it to daily values to be used on the Forecasting. 
This assigns the date of 2024 to the dim of time. need to change to differant years or repeat for multiple years.

"""

import xarray as xr
import pandas as pd
import numpy as np

ds = xr.open_dataset("drifter_monthlymeans.nc")

#  Rename dimension
ds = ds.rename({"ClimatologicalMonth": "time"})

#  Assign mid-month timestamps (length must equal 12)
clim_time = pd.date_range("2024-01-01", periods=12, freq="MS") + pd.Timedelta(days=14)
ds["time"] = clim_time   # <-- direct assignment avoids reindexing

# Create cyclic extension properly
jan_next = ds.isel(time=0).copy()
jan_next["time"] = jan_next.time + np.timedelta64(365, "D")
dec_prev = ds.isel(time = -1).copy()
dec_prev["time"] = dec_prev.time - np.timedelta64(365, "D")
ds_ext = xr.concat([dec_prev, ds, jan_next], dim="time")

#  Create daily target time
daily_time = pd.date_range("2024-01-01", "2024-12-31", freq="D")

#  Interpolate
ds_daily = ds_ext.interp(time=daily_time)
ds_daily = ds_daily.rename({"U": "uo", "V" : "vo"})



ds_daily.to_netcdf("drifter_climatology_daily_values.nc")



