"""
Run Forcasts for dFADs with dynamical CMEMS models istead of static velocity fields

Can run each month at a time since field for all dFADs is the same. 

1) Collect all dFADs initial possitions for forcasting.
        - Same method as 'run_model.py' 

2) make forecasts for entire period

3) 

"""
import parcels
import geopandas as gpd 
from pathlib import Path
import xarray as xr
from datetime import timedelta
import zarr
import functions.funcs as fad
import pandas as pd 
import numpy as np 
import shapely as sp 

fname = r"..\..\Data\cmems.nc" ### Change the field to cmems
field = xr.open_dataset(fname )


ds = gpd.read_parquet(r"..\..\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")

monthrange = pd.date_range("2024-01-1","2025-01-1", freq= "MS")
for month in range(len(monthrange)-1):
    startdate = monthrange[month]
    enddate = monthrange[month+1] + pd.Timedelta(days =7)

        ## Initalizing the final dateset of the dFADs
    dssave = pd.DataFrame() 
    dssave = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])

    Monthdaterange = pd.date_range(startdate, enddate - pd.Timedelta(days =7), freq= "D")

    dFADs = pd.DataFrame(columns = ["BuoyName","lat", "lon", "TimeStamp", "x_speed", "y_speed"] )

    for day in Monthdaterange[:-1]: ## all exepnt last because its the first day in the following month
        endofday = day + pd.Timedelta(days= 1)
        ds_active = fad.querry_date(ds, day) ## All of the active dFADs at this time  #863 total points 
        ds_active = ds_active.reset_index()
        columns = ["TimeStamp", "x_speed", "y_speed"]
        ds_locations = pd.DataFrame(columns = ["BuoyName","lat", "lon", "TimeStamp", "x_speed", "y_speed"])
        for label in columns: 
            longlist, ids = fad.Column_to_List(ds_active, label, idlist = True)
            
            ds_locations[label] = longlist
        lat, lon  = fad.list_of_latlon(ds_active, droplast= False)
        ds_locations["lat"] = lat
        ds_locations["lon"] =lon
        ds_locations["BuoyName"] = ids
        ds_locations.TimeStamp = pd.to_datetime(ds_locations.TimeStamp)
            
        ds_locations = ds_locations.query(f"TimeStamp > @day")
        ds_locations = ds_locations.query(f"TimeStamp < @endofday")

        ## remove duplicates
        ds_locations = ds_locations.drop_duplicates(subset=["lat"], keep="first")
        ds_locations = ds_locations.drop_duplicates(subset=["lon"], keep="first").reset_index(drop = True)
        ds_locations = ds_locations.sort_values('TimeStamp').groupby("BuoyName").first().reset_index()

        ## These are initial locations to make forecasts from
        dFADs = pd.concat([dFADs,ds_locations.reset_index()])
        dFADs["TimeStamp"] = pd.to_datetime(dFADs["TimeStamp"])
    ds_active = fad.querry_date_range(ds, startdate= startdate, enddate= enddate)
    dFADs = dFADs.reset_index(drop = True)
    dFADs.TimeStamp = pd.to_datetime(dFADs.TimeStamp)
    print(f"amount of dFADs: {len(dFADs)}")
    #_________________________________________
    ## Make the model... 
    print(f"making Forecasts from {startdate} -- {enddate}")

    ## Make the model... 
    filenames = {"uo": fname, "vo": fname}
    variables  = {"U": "uo", "V": "vo"}
    dimensions = {"lat": "latitude", "lon": "longitude", "time" : "time"}
    ## fix this and make it a non static field
    field_t = field.sel(time = slice(startdate, enddate), depth = 15.81007).drop_vars("time")## IF CMEMS add depth = 15 argument
    runtime = enddate - startdate + pd.Timedelta(days = 5)

    # fieldsetperm = parcels.FieldSet.from_netcdf(filenames, variables, dimensions)
    fieldset  = parcels.FieldSet.from_xarray_dataset(field_t, variables, dimensions, allow_time_extrapolation= True) 
    fieldset.add_constant("halo_west", fieldset.U.grid.lon[0])
    fieldset.add_constant("halo_east", fieldset.U.grid.lon[-1])
    fieldset.add_constant("halo_north", fieldset.U.grid.lat[-1])
    fieldset.add_constant("halo_south", fieldset.U.grid.lat[0])
    fieldset.add_periodic_halo(zonal = True , meridional= True)

    def boundryCondition(particle, fieldset,time):
        if particle.lon < fieldset.halo_west or particle.lon > fieldset.halo_east:
            particle.delete()
        if particle.lat < fieldset.halo_south or particle.lat > fieldset.halo_north:
            particle.delete()
            
    def Age(particle, fieldset, time):
        particle.age += particle.dt / 3600

    ### need to add to kill particles after 7 days. 

    def end_forcast(particle, fieldset, time): ## need to check if this is working correctly
        if particle.age > 7*24:
            particle.delete()
        
    dFADs["timedelta"] = (dFADs.TimeStamp - startdate).dt.total_seconds()

    Particles = parcels.JITParticle.add_variable("age", initial = 0) 
    Particles = Particles.add_variable("Buoyindex", to_write = 'once')

    pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                        lat = dFADs.lat.to_list() , time = dFADs.timedelta, Buoyindex = dFADs.index.values) 

    output_memorystore = zarr.storage.MemoryStore()
    output_file = pset.ParticleFile(name = f"..\output\TestParticleFile{month}.zarr", outputdt =timedelta(hours= 1))


    pset.execute([parcels.AdvectionRK4, boundryCondition, Age, end_forcast], 
                runtime = runtime, ##this should be 8 days 
                dt = timedelta(minutes =5),  ## change this 
                output_file = output_file, 
                )

    ####____________________________
    print("Model run complete \n handling the output")
    ###_____________________________
