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
import pandas as pd

def log(log_output : str):
    from functions.settings import LOG_FILE
    with open(LOG_FILE, "a") as log_file:
        log_file.write(log_output)

def Run_model_dynamical(startdate:pd.Timestamp, enddate:pd.Timestamp, monthindex:int, configfile:str, mute_warnings = True, progressbar = False):
     
    """Wrapper function to run cmems dynamical forecasts"""
    
    import parcels
    import geopandas as gpd 
    import xarray as xr
    from datetime import timedelta
    import zarr
    import pandas as pd 
    import numpy as np  
    import tomli as tomllib
    import functions.settings as settings


    from functions.parcels.Dataloader_alligner import Dataloader, Alligner
    from functions.parcels.kernels import persistence_AdvectionRK4, boundryCondition, Age, end_forcast
    ## MUTES all parcels warning 
    if mute_warnings == True: 
        import warnings
        import logging
        warnings.filterwarnings("ignore", category=UserWarning, module="parcels")
        logging.getLogger("parcels.tools.loggers").setLevel(logging.WARNING)
   
    with open(configfile, 'rb') as f:
        config = tomllib.load(f)
        
    ## set model Params from config file
    persistence = config['persistence']
    persistencewindow = config['persistence_window']
    usewinds= config['wind']
    filename = config['currents_file']
    depth = config['depth']
    forecast_length = pd.Timedelta(days = config['forecast_length'])
    stokes_drift = config['stokes_drift']
    
    #loading the data
    cmems = xr.open_dataset(settings.GLORYS_FILE)
    cmems = cmems.sel(time = slice(startdate, enddate+forecast_length), 
              depth = depth) 
    Uo = cmems.uo +cmems.vo*1j

    if stokes_drift:
        winds = xr.open_dataset(settings.ERA5_FILE)
        U_stokes  = winds.ust + winds.vst*1j
        Uo = Uo + U_stokes

    if config['bias'] == True: 
        m = (config['GLORYs_correction']['currents'][0] + 
             config['GLORYs_correction']['currents'][1]*1j)
        Uo_clim_mean = (config['GLORYs_correction']['Uo_clim_mean'][0] +
                        config['GLORYs_correction']['Uo_clim_mean'][1]*1j)
        U_dfad_mean  = (config['GLORYs_correction']['U_dfad_mean'][0] +
                        config['GLORYs_correction']['U_dfad_mean'][1]*1j)
        Uo = m*(Uo - Uo_clim_mean) + U_dfad_mean

    if usewinds == True:
        winds = xr.open_dataset(settings.ERA5_FILE)
        winds = winds.sel(time = slice(startdate, enddate+forecast_length))
        windsi = winds.interp_like(cmems) ## adding winds 
        ## Y = m*Uo + n*W
        W = windsi.uo +windsi.vo*1j
        n = (config['GLORYs_correction']['wind'][0] + 
             config['GLORYs_correction']['wind'][1]*1j)
        W_clim_mean = (config['GLORYs_correction']['W_clim_mean'][0] +
                       config['GLORYs_correction']['W_clim_mean'][1]*1j)
        Uo = Uo + n * (W - W_clim_mean)


    cmems['uo'] = Uo.real
    cmems['vo'] = Uo.imag
    field = cmems


    ## loads the dFAD data
    ds = gpd.read_parquet(settings.dFAD_DATA)
    loader = Dataloader(ds)
    loader.Firstposstions_multidays(
        startdate=startdate,
        enddate=enddate,
        window_length=pd.Timedelta(days=1),
        persistencewindow = persistencewindow
    )
    dFADs = loader.dFADs.reset_index(drop=True)
    dFADs["TimeStamp"] = pd.to_datetime(dFADs["TimeStamp"])

    # Encode BuoyName as a stable integer using globally sorted unique names.
    all_buoys = sorted(ds["BuoyName"].unique())
    buoy_to_int = {name: idx for idx, name in enumerate(all_buoys)}
    dFADs["BuoyIntID"] = dFADs["BuoyName"].map(buoy_to_int)

    print(f"{monthindex} amount of dFADs: {len(dFADs)}")
    #_________________________________________
    ## Make the model... 
    print(f"{monthindex} making Forecasts from {startdate} -- {enddate}")

    ## Make the model... 
    variables  = {"U": "uo", "V": "vo"}
    dimensions = {"lat": "latitude", "lon": "longitude", "time" : "time"}
    ## fix this and make it a non static field
    field_t = field
    runtime = enddate - startdate + pd.Timedelta(days = 7)

    # fieldsetperm = parcels.FieldSet.from_netcdf(filenames, variables, dimensions)
    fieldset  = parcels.FieldSet.from_xarray_dataset(field_t, 
                                                     variables, dimensions, allow_time_extrapolation= True) 
    fieldset.add_constant("halo_west", fieldset.U.grid.lon[0])
    fieldset.add_constant("halo_east", fieldset.U.grid.lon[-1])
    fieldset.add_constant("halo_north", fieldset.U.grid.lat[-1])
    fieldset.add_constant("halo_south", fieldset.U.grid.lat[0])
    fieldset.add_periodic_halo(zonal = True , meridional= True)

        
    dFADs["timedelta"] = (dFADs.TimeStamp - startdate).dt.total_seconds()

    Particles = parcels.ScipyParticle.add_variable("age", initial = 0) 
    Particles = Particles.add_variable("Buoyindex", to_write = 'once')

    pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                        lat = dFADs.lat.to_list() , time = dFADs.timedelta.to_list(), Buoyindex = dFADs["BuoyIntID"].values) 
    
    if persistence == True:
        Particles = Particles.add_variable("ui", to_write = 'once') ## units of degree/second
        Particles = Particles.add_variable("vi", to_write = 'once') ## units od degree/second
        Particles = Particles.add_variable("tau",initial = 0.83*24,to_write = 'once') ## units of hours
        pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                        lat = dFADs.lat.to_list() , time = dFADs.timedelta, 
                        Buoyindex = dFADs["BuoyIntID"].values, 
                        ui = dFADs.x_speed_prev/1000/111, vi = dFADs.y_speed_prev/1000/111)
        output_memorystore = zarr.storage.MemoryStore()
        output_file = pset.ParticleFile(name=output_memorystore, outputdt=timedelta(hours=1))
        pset.execute([persistence_AdvectionRK4, boundryCondition, Age, end_forcast], 
                        runtime = runtime, ##this should be 8 days 
                        dt = timedelta(minutes =20), 
                        output_file = output_file,
                        verbose_progress= progressbar 
                        )
    else: 
        pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                        lat = dFADs.lat.to_list() , time = dFADs.timedelta, 
                        Buoyindex = dFADs["BuoyIntID"].values)
        output_memorystore = zarr.storage.MemoryStore()
        output_file = pset.ParticleFile(name=output_memorystore, outputdt=timedelta(hours=1))
        pset.execute([parcels.AdvectionRK4, boundryCondition, Age, end_forcast], 
                    runtime = runtime, ##this should be 8 days 
                    dt = timedelta(minutes =20), 
                    output_file = output_file, 
                    verbose_progress= progressbar
                    )
    ## save output log to txt file 
    ####____________________________
    log(f"Model {monthindex} {startdate} {enddate} run complete\n")
    ###_____________________________

    output = xr.open_zarr(output_memorystore)

    engine = Alligner(ds)
    engine.allign_data_MultipleDays(output, startdate, enddate,
                                    pd.Timedelta(days = 7), monthindex)
    engine.dssave.to_csv(settings.CORE_DIR / 'scripts' / 'forecasts'/ 'output' / f'Forecast{[monthindex]}.csv')
    output_memorystore.clear()

    log(f"Model {monthindex} {startdate} {enddate} output saved\n")



