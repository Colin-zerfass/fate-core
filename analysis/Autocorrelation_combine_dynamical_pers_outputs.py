
# ### Autocolitation model
# Model that can combined Percistence(past memory of the particles trajectories) with a dynamical model once the velocities are not corrilated. 
# $\tau$ was calcuated from Code\Autocorrelation: this calcuated the correlation between past times and future velcities and is going to be used as the decorrilation timescale of the persitence model. 
# 
# what is needed: 
# - Percistence forcasts 
# - Dynamical model (ie: cmems, Krigging, Whatever other model that is more accurate on long time horizons)
# - $\tau$ : 0.84 days
# 
# Errors: 
# - Cmems and persitence are not the same length before and after because cmems could leave the box.
# - Not the same amount of rows Before an after the merge(more after merge), Where do these come from?? 

 
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np 
dsp = pd.read_csv(r"saved_output\combined_Percistance_2022_2024.csv")
dsc = pd.read_csv(r"saved_output\intial_speed_dif_OSCAR_CMEMS_wind_2022_2025.csv")
dsp = dsp.rename(columns={"DateTime": "Time", "Latitude_true": "lat_true", "Longitude_true": "lon_true", "lead_time_hours":"leadtime"
                          ,"Latitude_persistence": "lat_pers", "Longitude_persistence": "lon_pers"})
dsc = dsc.rename(columns={"lat_forcast": "lat_cmems", "lon_forcast": "lon_cmems"})
print(dsc.columns)
print(dsp.columns)
print(len(dsc), len(dsp))

 
dsc["lat_cmems"] = dsc["lat_cmems"].where(~dsc["lat_cmems"].isna(), dsc["lat_true"] ) ## replaces 0 leadtimes with the true values
dsc["lon_cmems"] = dsc["lon_cmems"].where(~dsc["lon_cmems"].isna(), dsc["lon_true"] ) 

 
ds = pd.merge(dsp, dsc, on=["BuoyID","Time","leadtime"], how="inner", suffixes=("_pers","_cmems"))
ds["Allign_error"] = ds["lat_true_pers"] - ds["lat_true_cmems"] + ds["lon_true_pers"] - ds["lon_true_cmems"]
ds["leadtime"] = pd.to_timedelta(ds["leadtime"], unit = "h").dt.round("1min")
ds["Time"] = pd.to_datetime(ds["Time"])
ds["StartTime"] = (ds["Time"] - ds["leadtime"]).dt.round("1min")

 
## If they are in order of leadtimes for each startime which they are...? Then can just subtract from previosois point.
## this works for all cases except when they leadtime = 0 but those should have no delta so can fix those after. 
## Make sure they are sorted correctly
ds = ds.sort_values(["BuoyID", "StartTime","leadtime"]).copy()
groups = ds.groupby(["BuoyID", "StartTime"])
ds["Delta_pers"]= groups["lon_pers"].diff() + 1j*groups["lat_pers"].diff()
ds["Delta_cmems"] = groups["lon_cmems"].diff() + 1j*groups["lat_cmems"].diff()

 
"""Now make a new forcast based using the persistence decorilaton Timescale
1) make Column that is weight of percistence"""

Tau = 0.83*24 # hrs

lead_hours = ds["leadtime"].dt.total_seconds() / 3600.0
ds["pers_weight"] = np.exp(- lead_hours / Tau)

ds["Delta"] = ds["pers_weight"]*ds["Delta_pers"] + (1-ds["pers_weight"] )*ds["Delta_cmems"]


# compute complex initial position (use the *_pers true columns created by the merge)
initial_complex = ds["lon_true_pers"] + 1j * ds["lat_true_pers"]

# replace NaNs in Delta with the initial complex position, then groupwise cumsum
ds["Delta_filled"] = ds["Delta"].where(ds["Delta"].notna(), initial_complex) ##this has to be only NaN deltas are at t = 0, 
ds["combined"] = ds.groupby(["BuoyID", "StartTime"])["Delta_filled"].transform(lambda x: x.cumsum())

 
## cleaning the dataframe and renaming into form of other forcasts
print(ds.columns)
# optional: drop helper column
ds_final = ds.drop(columns=["Delta_filled","Unnamed: 0_pers",'lat_pers', 'lon_pers', 'speed_ms_persistence','Unnamed: 0_cmems', 'lat_true_cmems', 'lon_true_cmems', 'lat_cmems',
       'lon_cmems', 'Allign_error','pers_weight'])

ds_final = ds_final.rename(columns={"lat_true_pers": "lat_true", "lon_true_pers": "lon_true"}).reset_index(drop = True)
ds_final["lat_forcast"] = np.imag(ds.combined)
ds_final["lon_forcast"] = np.real(ds.combined)
ds_final["leadtime"] = ds_final["leadtime"].dt.total_seconds()/3600

ds_final.to_csv("saved_output/Autocorrelation_intial_speed_dif_OSCAR_CMEMS_wind_2022_2025.csv")


