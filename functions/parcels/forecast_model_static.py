r"""from DIR of "Code\parcels" run this file. 
to change from cmems to oscar
1) remove depth = 15 when sellecting the velocity field 
2) add Transpose = True when loading the field into the velocity field. OSCAR has coords of (lon, lat)... 
3) Change variable from uo, vo to u,v

fix drop var= time in line 169
"""

import pandas as pd
from functions.parcels.forecast_model_dynamic import log 

def Run_model_static(startdate:pd.Timestamp, enddate:pd.Timestamp, monthindex:int, configfile:str, mute_warnings = True, progressbar = False):
    import parcels
    import geopandas as gpd 
    import pandas as pd
    import xarray as xr
    from datetime import timedelta
    import zarr
    import tomli as tomllib 
    import numpy as np

    from functions.funcs import querry_date_range
    from functions.parcels.Dataloader_alligner import Dataloader, Alligner
    from functions.parcels.kernels import persistence_AdvectionRK4, boundryCondition, Age, end_forcast
    import functions.settings as settings

    if mute_warnings == True:
        import warnings
        import logging
        warnings.filterwarnings("ignore", category=UserWarning, module="parcels")
        logging.getLogger("parcels.tools.loggers").setLevel(logging.WARNING)    #def Run_model(startmonth:pd.Timestamp, endmonth:pd.Timestamp, monthindex:int, configfile:str, log_lock=None): 

    #load config file
    print(configfile)
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
    ##_______________________
    # load Currents and Winds 
    ##_______________________


    if filename == "cmems":
        cmems = xr.open_dataset(settings.GLORYS_FILE)
        cmems = cmems.sel(time = slice(startdate, enddate+forecast_length), 
              depth = depth) 
        Uo = cmems.uo +cmems.vo*1j
        if stokes_drift:
            winds = xr.open_dataset(settings.ERA5_FILE)
            U_stokes  = winds.ust + winds.vst*1j
            Uo = Uo + U_stokes

        if config['bias'] == True:
            m = config['GLORYs_correction']['currents'][0] + config['GLORYs_correction']['currents'][1]*1j
            Uo = m*Uo

        if usewinds == True:
            winds = xr.open_dataset(settings.ERA5_FILE)
            winds = winds.sel(time = slice(startdate, enddate+forecast_length))
            n = config['GLORYs_correction']['wind'][0] + config['GLORYs_correction']['wind'][1]*1j
            windsi = winds.interp_like(cmems)
            W = windsi.uo +windsi.vo*1j
            Uo = Uo + n*W
            
        cmems['uo'] = Uo.real
        cmems['vo'] = Uo.imag
        field = cmems

    if filename == "OSCAR":
        oscar = xr.open_dataset(settings.OSCAR_FILE)
        oscar = oscar.sel(time = slice(startdate, enddate+forecast_length))
        Uo = oscar.uo +oscar.vo*1j
        if stokes_drift:
            winds = xr.open_dataset(settings.ERA5_FILE)
            U_stokes  = winds.ust + winds.vst*1j
            Uo = Uo + U_stokes

        if config['bias'] == True:
            m = config['OSCAR_correction']['currents'][0] + config['OSCAR_correction']['currents'][1]*1j
            Uo = m*Uo

        if usewinds == True:
            winds = xr.open_dataset(settings.ERA5_FILE)
            winds = winds.sel(time = slice(startdate, enddate+forecast_length))
            windsi = winds.interp_like(oscar)
            n = config['OSCAR_correction']['wind'][0] + config['OSCAR_correction']['wind'][1]*1j
        
            W = windsi.uo +windsi.vo*1j
            Uo = Uo + n*W

        oscar['uo'] = Uo.real
        oscar['vo'] = Uo.imag
        field = oscar

    ## WARNING climatology OUTdated assing fname to field like OSCAR and CMEMS above
    if filename == "climatological": 
        ## WARNING The dates in climatological Dataset are from 2024-2025, need to change dates if running outside of those dates. 
        fname = rf"..\Data\drifter_climatology_daily_values.nc"

    ds = gpd.read_parquet(settings.dFAD_DATA)
    ds_timerange = querry_date_range(ds,startdate , enddate).reset_index(drop = True)
    daterange = pd.date_range(startdate, enddate)
    daily_stores = []
    ## developing an int from the orriginal dataframe
    all_buoys = sorted(ds["BuoyName"].unique())
    buoy_to_int = {name: idx for idx, name in enumerate(all_buoys)}

    for day in daterange:
        target_date = day ## picks dFAD locations one day after this date 
    ##_______________
    ##Loads the dFADs 
    ##_______________

        loader = Dataloader(ds_timerange)
        loader.First_possitions(day, persistencewindow= persistencewindow)
        dFADs = loader.dFADs.reset_index(drop = True)
        dFADs['TimeStamp'] = pd.to_datetime(dFADs['TimeStamp'])
        
        dFADs["BuoyIntID"] = dFADs["BuoyName"].map(buoy_to_int)
    ##_______________
    ##Make the Model
    ##_______________

        variables  = {"U": "uo", "V": "vo"}
        dimensions = {"lat": "latitude", "lon": "longitude", "time": "time"}
        # Select a single time snapshot for this day so the field is static
        # throughout the 8-day execute (no time interpolation, no reshape issues).
        field_day = field.sel(time=[day], method='nearest')
        fieldset  = parcels.FieldSet.from_xarray_dataset(field_day, variables, dimensions, allow_time_extrapolation= True) 

        fieldset.add_constant("halo_west", fieldset.U.grid.lon[0])
        fieldset.add_constant("halo_east", fieldset.U.grid.lon[-1])
        fieldset.add_constant("halo_north", fieldset.U.grid.lat[-1])
        fieldset.add_constant("halo_south", fieldset.U.grid.lat[0])
        fieldset.add_periodic_halo(zonal = True , meridional= True)

        dFADs["timedelta"] = (dFADs.TimeStamp - target_date).dt.total_seconds()

        Particles = parcels.ScipyParticle.add_variable("age", initial = 0) 
        Particles = Particles.add_variable("Buoyindex", to_write = 'once')
    
    ##__________
    ## Run Model
    ##__________

        if persistence == True:
            Particles = Particles.add_variable("ui", to_write = 'once') ## units of degree/second
            Particles = Particles.add_variable("vi", to_write = 'once') ## units od degree/second
            Particles = Particles.add_variable("tau",initial = 0.83*24,to_write = 'once') ## units of hours
            pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                            lat = dFADs.lat.to_list() , time = dFADs.timedelta, Buoyindex = dFADs["BuoyIntID"].values, 
                                            ui = dFADs.x_speed_prev/1000/111, vi = dFADs.y_speed_prev/1000/111) 
            output_memorystore = zarr.storage.MemoryStore()
            output_file = pset.ParticleFile(name = output_memorystore, outputdt =timedelta(hours= 1))

            pset.execute([persistence_AdvectionRK4, boundryCondition, Age, end_forcast], 
                            runtime = timedelta(days = 8), ##this should be 8 days 
                            dt = timedelta(minutes =20), 
                            output_file = output_file, 
                            verbose_progress= progressbar
                            )
        else: 
            pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                            lat = dFADs.lat.to_list() , time = dFADs.timedelta, Buoyindex = dFADs["BuoyIntID"].values) 
            output_memorystore = zarr.storage.MemoryStore()
            output_file = pset.ParticleFile(name = output_memorystore, outputdt =timedelta(hours= 1))
            pset.execute([parcels.AdvectionRK4, boundryCondition, Age, end_forcast], 
                        runtime = timedelta(days = 8), ##this should be 8 days 
                        dt = timedelta(minutes =20), 
                        output_file = output_file, 
                        verbose_progress= progressbar
                        )

        daily_stores.append(output_memorystore)

    ##_______________
    ## Saving to CSV
    ##________________
    # Combine all daily zarr stores into one before aligning.
    # Open with decode_times=False so times are kept as raw integers,
    # avoiding the CF datetime re-encoding bug on to_zarr.
    daily_ds = []
    offset = 0
    for s in daily_stores:
        ds_day = xr.open_zarr(s)
        n = ds_day.sizes['trajectory']
        # Re-index so trajectory IDs are unique across days; without this,
        # every run starts at 0 and xr.concat aligns on the coordinate,
        # merging particles from different days with the same ID.
        ds_day = ds_day.assign_coords(trajectory=np.arange(offset, offset + n))
        daily_ds.append(ds_day)
        offset += n
    combined_ds = xr.concat(daily_ds, dim='trajectory')

    engine = Alligner(ds)
    engine.allign_data_MultipleDays(combined_ds, startdate, enddate,
                                    pd.Timedelta(days=7), monthindex)
    engine.dssave.to_csv(settings.CORE_DIR / 'scripts' / 'forecasts'/ 'output' / f'Forecast{[monthindex]}.csv')
    for s in daily_stores:
        s.clear()

    log(f"Model {monthindex} {startdate} {enddate} run complete\n")

