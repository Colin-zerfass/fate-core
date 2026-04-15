"""
python Dynamical_cmems_monthly_parallel.py 

Run Forcasts for dFADs with dynamical CMEMS models instead of static velocity fields

Can run each month at a time since the months are independent  

1) Collect all dFADs initial possitions for forcasting.
        - Same method as 'run_model.py' 

2) make forecasts for entire 1 month period

3) Save output to a .zar, this is not alligned with to the actual dFAD data
have to run Dataloader.py afterwards

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
import sys 
import tomllib

from functions.Dataloader_alligner import Dataloader
from functions.persistence_AdvectionRK4 import persistence_AdvectionRK4

def Run_model(startdate, enddate, monthindex, configfile):
     
    """Wrapper function to run cmems dynamical forecasts"""

    with open(configfile, 'rb') as f:
        config = tomllib.load(f)
        
    ## set model Params from config file
    persistence = config['persistence']
    persistencewindow = config['persistence_window']
    usewinds= config['wind']
    filename = config['currents_file']
    depth = config['depth']
    forecast_length = pd.Timedelta(days = config['forecast_length'])
    
    #loading the data
    cmems = xr.open_dataset(config['GLORYs_data'])
    cmems.sel(time = slice(startdate, enddate+forecast_length), 
              depth = depth) ### Change the field to cmems

    if usewinds == True:
        winds = xr.open_dataset(config['Wind_data'])
        winds.sel(time = slice(startdate, enddate+forecast_length))
        windsi = winds.interp_like(cmems) ## adding winds 
        ## Y = m*Uo + n*W
        m = (config['GLORYs_correction']['currents'][0] + 
             config['GLORYs_correction']['currents'][1]*1j)
        n = (config['GLORYs_correction']['wind'][0] + 
             config['GLORYs_correction']['wind'][1]*1j)
        Uo = cmems.uo +cmems.vo*1j
        W = windsi.uo +windsi.vo*1j
        Y = m*Uo + n*W
        cmems['uo'] = Y.real
        cmems['vo'] = Y.imag
    field = cmems.sel(depth = depth, method = "nearest")


    ## loads the dFAD data
    ds = gpd.read_parquet(config['dFAD_data'])
    loader = Dataloader(ds, persistencewindow = persistencewindow)
    loader.Firstposstions_multidays(
        startdate=startdate,
        enddate=enddate,
        window_length=pd.Timedelta(days=1),
    )
    dFADs = loader.dFADs.reset_index(drop=True)
    dFADs["TimeStamp"] = pd.to_datetime(dFADs["TimeStamp"])

    print(f"{monthindex} amount of dFADs: {len(dFADs)}")
    #_________________________________________
    ## Make the model... 
    print(f"{monthindex} making Forecasts from {startdate} -- {enddate}")

    ## Make the model... 
    variables  = {"U": "uo", "V": "vo"}
    dimensions = {"lat": "latitude", "lon": "longitude", "time" : "time"}
    ## fix this and make it a non static field
    field_t = field.sel(time = slice(startdate, enddate + pd.Timedelta(days= 7)), depth = depth) ## IF CMEMS add depth = 15 argument
    runtime = enddate - startdate + pd.Timedelta(days = 7)

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
                                        lat = dFADs.lat.to_list() , time = dFADs.timedelta.to_list(), Buoyindex = dFADs.index.values) 

    output_memorystore = zarr.storage.MemoryStore()
    output_file = pset.ParticleFile(name = f"..\output\TestParticleFile{monthindex}.zarr", outputdt =timedelta(hours= 1))


    if persistence == True:
        Particles = Particles.add_variable("ui", to_write = 'once') ## units of degree/second
        Particles = Particles.add_variable("vi", to_write = 'once') ## units od degree/second
        Particles = Particles.add_variable("tau",initial = 0.83*24,to_write = 'once') ## units of hours
        pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                        lat = dFADs.lat.to_list() , time = dFADs.timedelta, 
                                        Buoyindex = dFADs.BuoyName.index, 
                                        ui = dFADs.x_speed_prev/1000/111, vi = dFADs.y_speed_prev/1000/111) 

        pset.execute([persistence_AdvectionRK4, boundryCondition, Age], 
                        runtime = runtime, ##this should be 8 days 
                        dt = timedelta(minutes =20), 
                        output_file = output_file, 
                        )
    else: 
        pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                        lat = dFADs.lat.to_list() , time = dFADs.timedelta, 
                                        Buoyindex = dFADs.BuoyName.index) 
        pset.execute([parcels.AdvectionRK4, boundryCondition, Age], 
                    runtime = runtime, ##this should be 8 days 
                    dt = timedelta(minutes =20), 
                    output_file = output_file, 
                    )
    ## save output log to txt file 
    ####____________________________
    with open(r"..\output\Output_logs.txt", "a") as log_file:
        log_file.write(f"Model {monthindex} {startdate} {enddate} run complete\n")
    ###_____________________________


## run models
if __name__ == "__main__":  
    import multiprocessing as mp
    import sys 
    import itertools
    
    """Method of running the model on given number of threads, one model runs on each thread sectioned by the monthrange above"""
    ### could work on running for less than a one month period
    

    config_name = sys.argv[1]
    with open(config_name, 'rb') as f:
        config = tomllib.load(f)

    totalstartdate = config['startdate']
    totalenddate = config['enddate']
    print(totalstartdate)
    print(totalenddate)
    with open(r"..\output\Output_logs.txt", "a") as log_file:
        log_file.write(f"Starting Run: {config['output_name']}")
            
    monthrange = pd.date_range(totalstartdate, totalenddate, freq="MS")

    # Build tuples of (start, end, index)
    inputs = list(zip(
        monthrange[:-1],
        monthrange[1:],
        range(len(monthrange)-1), ## need to revert this back to range(len(monthrange)-1)
        itertools.repeat(config_name, len(monthrange)-1),
    ))

    with mp.Pool(processes=config['parallel_cores']) as pool:
        results = pool.starmap(Run_model, inputs)

    print("Results:", results)

