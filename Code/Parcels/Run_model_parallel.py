r"""from DIR of "Code\parcels" run this file. 
to change from cmems to oscar
1) remove depth = 15 when sellecting the velocity field 
2) add Transpose = True when loading the field into the velocity field. OSCAR has coords of (lon, lat)... 
3) Change variable from uo, vo to u,v

fix drop var= time in line 169
"""

import parcels
import geopandas as gpd 
import os
import sys
from pathlib import Path
import xarray as xr
from datetime import timedelta
import zarr
#sys.path.insert(0, str(Path.cwd().parent))
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.append(parent_dir)
import tomllib

from functions.funcs import *

def persistence_AdvectionRK4(particle, fieldset, time):  # pragma: no cover
    """Advection of particles using fourth-order Runge-Kutta integration.
    with an added persitence model 
    units  WARNING
    particle.dt : (Seconds),
    particle_dlon/dlat : degrees
    particle.age & particle.tau (hours), 
    """
    import numpy as np 
    (u1, v1) = fieldset.UV[particle]
    lon1, lat1 = (particle.lon + u1 * 0.5 * particle.dt, particle.lat + v1 * 0.5 * particle.dt)
    (u2, v2) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat1, lon1, particle]
    lon2, lat2 = (particle.lon + u2 * 0.5 * particle.dt, particle.lat + v2 * 0.5 * particle.dt)
    (u3, v3) = fieldset.UV[time + 0.5 * particle.dt, particle.depth, lat2, lon2, particle]
    lon3, lat3 = (particle.lon + u3 * particle.dt, particle.lat + v3 * particle.dt)
    (u4, v4) = fieldset.UV[time + particle.dt, particle.depth, lat3, lon3, particle]
    advection_dlon = (u1 + 2 * u2 + 2 * u3 + u4) / 6.0 * particle.dt  # noqa
    advection_dlat = (v1 + 2 * v2 + 2 * v3 + v4) / 6.0 * particle.dt  # noqa

    ## Calculating persistence 
    persistence_dlon = particle.ui*particle.dt
    persistence_dlat = particle.vi*particle.dt

    # Weighting how much persistence to use
    persistence_frac = np.exp(-particle.age/particle.tau)
    if particle.age < 4*particle.tau: 
        #print(particle.dt, particle.ui, persistence_frac)
        persistence_frac = np.exp(-particle.age/particle.tau)
    else: 
        persistence_frac = 0

    # final displacement 
    particle_dlon += persistence_dlon*persistence_frac + advection_dlon*(1- persistence_frac)
    particle_dlat += persistence_dlat*persistence_frac + advection_dlat*(1- persistence_frac)


def Run_model(startmonth:pd.Timestamp, endmonth:pd.Timestamp, monthindex:int, configfile:str):
    #load config file
    print(configfile)
    with open(configfile, 'rb') as f:
        config = tomllib.load(f)

    ## set model Params from config file
    persistence = config['persistence']
    persistencewindow = config['persistence_window']
    usewinds= config['wind']
    print(usewinds)
    filename = config['currents_file']
    depth = config['depth']

    ##_______________________
    # load Currents and Winds 
    ##_______________________

    winds = xr.open_dataset(r"..\Data\ERA5_10m_winds.nc")
    winds = winds.rename({"lat" :'latitude', 'lon': 'longitude'})
    winds = winds.sortby('latitude')

    if filename == "cmems":
        cmems = xr.open_dataset(rf"..\Data\cmems.nc")
        if usewinds == True:
            windsi = winds.interp_like(cmems)
            m = 7.73709253e-01+1.62143540e-01j
            n = 7.85629581e-03+1.45730160e-03j
            Uo = cmems.uo +cmems.vo*1j
            W = windsi.uo +windsi.vo*1j
            Y = m*Uo + n*W
            cmems['uo'] = Y.real
            cmems['vo'] = Y.imag
        field = cmems.sel(depth = depth, method = "nearest")

    if filename == "OSCAR":
        oscar = xr.open_dataset(rf"..\Data\OSCAR_combined_2021_2025v2.nc")
        oscar = oscar.rename({'lon':'longitude', 'lat':'latitude', 'u' : 'uo', 'v':'vo' })
        oscar = oscar.set_index(longitude='longitude', latitude='latitude')
        oscar = oscar.transpose('time' ,'latitude', 'longitude')
        if usewinds == True:
            print('using winds')
            windsi = winds.interp_like(oscar)
            m = 9.20652565e-01+1.94924156e-02j
            n = 1.05991333e-02+7.39510759e-03j
            Uo = oscar.uo +oscar.vo*1j
            W = windsi.uo +windsi.vo*1j
            Y = m*Uo + n*W
            oscar['uo'] = Y.real
            oscar['vo'] = Y.imag
        field = oscar

    ## WARNING climatology OUTdated assing fname to field like OSCAR and CMEMS above
    if filename == "climatological": 
        ## WARNING The dates in climatological Dataset are from 2024-2025, need to change dates if running outside of those dates. 
        fname = rf"..\Data\drifter_climatology_daily_values.nc"

    ds = gpd.read_parquet(r"..\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")
    daterange = pd.date_range(startmonth, endmonth)
    dssave = pd.DataFrame()
    dssave = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])
    for day in daterange:
        target_date = day ## picks dFAD locations one day after this date 

    ##_______________
    ##Loads the dFADs 
    ##_______________

        ds_active = querry_date(ds, date = target_date) ## All of the active dFADs at this time 
        ds_active = ds_active.reset_index()
        columns = ["TimeStamp", "x_speed", "y_speed"]
        ds_locations = pd.DataFrame()
        for label in columns: 
            longlist, ids = Column_to_List(ds_active, label, idlist = True)
            
            ds_locations[label] = longlist
        lat, lon  = list_of_latlon(ds_active, droplast= False)
        ds_locations["lat"] = lat
        ds_locations["lon"] =lon
        ds_locations["BuoyName"] = ids
        ds_locations.TimeStamp = pd.to_datetime(ds_locations.TimeStamp)
        if persistence == True: 
            ds_locations["x_speed_prev"] = (ds_locations.groupby("BuoyName")["x_speed"].transform(lambda x: x.rolling(window=persistencewindow, min_periods=1).mean())) ## Calcuates persistence based pervious window
            ds_locations["y_speed_prev"] = (ds_locations.groupby("BuoyName")["y_speed"].transform(lambda x: x.rolling(window=persistencewindow, min_periods=1).mean()))

        ##Filter Timestep by certain threshhold to get locations of FADS within closes  
        ## UPDATE:This might be better to interp these onto the specific time. 
        hourlim = 24
        time_threshhold  = pd.Timedelta(hours= hourlim)
        time_upper  = target_date + time_threshhold ## This is set for dFADs one day after the date 
        time_lower = target_date 
        ds_locations = ds_locations.query(f"TimeStamp > @time_lower")
        ds_locations = ds_locations.query(f"TimeStamp < @time_upper")
        #print(f"Amount of sampled dFAD within {hourlim} hrs : {len(ds_locations)}")
        ds_locations = ds_locations.drop_duplicates(subset=["lat"], keep="first")
        ds_locations = ds_locations.drop_duplicates(subset=["lon"], keep="first").reset_index(drop = True)

        ## New get only the first point of the day for the forcast.

        dFADs = ds_locations.sort_values('TimeStamp').groupby("BuoyName").first()
        print(f"{target_date}:Number of Unique dFADs/ points avalable: {len(dFADs)} ")
        if len(dFADs) < 1: 
            continue
        dFADs = dFADs.reset_index()

    ##_______________
    ##Make the Model
    ##_______________

        variables  = {"U": "uo", "V": "vo"}
        dimensions = {"lat": "latitude", "lon": "longitude"}
        if config['currents_file'] == 'cmems':
            field_t = field.sel(time = target_date, method = "nearest").drop_vars('time')
        if config['currents_file'] == 'OSCAR':
            field_t = field.sel(time = target_date, method = "nearest")
        fieldset  = parcels.FieldSet.from_xarray_dataset(field_t, variables, dimensions, allow_time_extrapolation= True) 

        fieldset.add_constant("halo_west", fieldset.U.grid.lon[0])
        fieldset.add_constant("halo_east", fieldset.U.grid.lon[-1])
        fieldset.add_constant("halo_north", fieldset.U.grid.lat[-1])
        fieldset.add_constant("halo_south", fieldset.U.grid.lat[0])
        fieldset.add_periodic_halo(zonal = True , meridional= True)
        print(f"fieldset lon: {fieldset.U.grid.lon[0]}")
        print(f"fieldset lat: {fieldset.U.grid.lat[0]}")
        def boundryCondition(particle, fieldset,time):
            if particle.lon < fieldset.halo_west or particle.lon > fieldset.halo_east:
                particle.delete()
            if particle.lat < fieldset.halo_south or particle.lat > fieldset.halo_north:
                particle.delete()
                
        def Age(particle, fieldset, time):
            particle.age += particle.dt / 3600
            
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
                                            lat = dFADs.lat.to_list() , time = dFADs.timedelta, Buoyindex = dFADs.BuoyName.index, 
                                            ui = dFADs.x_speed_prev/1000/111, vi = dFADs.y_speed_prev/1000/111) 
            output_memorystore = zarr.storage.MemoryStore()
            output_file = pset.ParticleFile(name = output_memorystore, outputdt =timedelta(hours= 1))

            pset.execute([persistence_AdvectionRK4, boundryCondition, Age], 
                            runtime = timedelta(days = 8), ##this should be 8 days 
                            dt = timedelta(minutes =5), 
                            output_file = output_file, 
                            )
        else: 
            pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                            lat = dFADs.lat.to_list() , time = dFADs.timedelta, Buoyindex = dFADs.BuoyName.index) 
            output_memorystore = zarr.storage.MemoryStore()
            output_file = pset.ParticleFile(name = output_memorystore, outputdt =timedelta(hours= 1))
            pset.execute([parcels.AdvectionRK4, boundryCondition, Age], 
                        runtime = timedelta(days = 8), ##this should be 8 days 
                        dt = timedelta(minutes =5), 
                        output_file = output_file, 
                        )

        ##_______________
        ## Saving to CSV
        ##________________

        buoy_list = dFADs.BuoyName.tolist() 
        ds_filtered = ds_active[ds_active["BuoyName"].isin(buoy_list)].reset_index(drop = True)


        def Forcast_snippit(ds: gpd.GeoDataFrame, dates, startdate, length)-> gpd.GeoDataFrame: 
            """Grad only the snipbit of dFAD trajectory that lines up with forcast window"""
            ds_s = ds.copy()
            forecast_end = startdate + length
            for i in range(len(ds)): ## Try and grab at the exact start times from dates they should be the same
                timelist = (ds_s.at[i,"TimeStamp"])
                timelist = pd.to_datetime(timelist)
                mask = (timelist >=startdate) & (timelist <= forecast_end)
                timelist = timelist[mask]
                coords = np.asarray(ds.at[i,"geometry"].coords)
                filtered_coords = coords[mask]
                ds_s.at[i,"TimeStamp"] = timelist
                if len(filtered_coords) > 1:
                    ds_s.at[i,"geometry"] = sp.geometry.LineString(filtered_coords)
                else: 
                    ds_s.at[i,"geometry"] = None
            return ds_s

        ds_short_t = Forcast_snippit(ds_filtered, dFADs.TimeStamp, target_date, pd.Timedelta(days = 8))


        ##Saving to CSV file
        output = xr.open_zarr(output_memorystore)
        dsout = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])
        masklarge =~ np.isnan(output.lat[:,:].values)
        dFADs_s = dFADs.iloc[output.Buoyindex.values]

        order = dFADs_s.BuoyName.tolist()
        order_map = {name: i for i, name in enumerate(order)}
        mask = ds_short_t["BuoyName"].isin(order)
        ds_short_ts = (
            ds_short_t[mask]
            .assign(order_index=lambda df: df["BuoyName"].map(order_map))
            .sort_values("order_index")
            .drop(columns="order_index")
            .reset_index(drop=True)
        )
        #Interpolating Forecast locations onto the true locations
        for i, index in enumerate(output.Buoyindex.values): 
            id = dFADs.BuoyName[index]
            row=  ds_short_ts.iloc[i]
            Times= row["TimeStamp"]
            starttime = Times[0]
            dFAD_leadtime = (Times - starttime).total_seconds() #.astype("int64")/1e9 ## convets leadtime to seconds
            mask = masklarge[i,:] 
            leadtimes = output.age[i,:].values[mask]*3600 ## converts to seconds also. 
            lats = output.lat[i,:].values[mask]
            lons = output.lon[i,:].values[mask]
            lat_interp = np.interp(dFAD_leadtime[1:], leadtimes,lats)
            lon_interp = np.interp(dFAD_leadtime[1:], leadtimes, lons) 
            lat_interp = np.insert(lat_interp, 0,np.nan)
            lon_interp = np.insert(lon_interp, 0,np.nan)
            if row["geometry"] == None:
                continue
            lon_true, lat_true= row["geometry"].xy
            Bouylist = [id]*len(lat_true) 
            dstemp= pd.DataFrame({"BuoyID": Bouylist, "Time": Times,
                                   "lat_true": lat_true,"lon_true":lon_true, ""
                                   "lat_forcast": lat_interp, "lon_forcast": lon_interp, 
                                   "leadtime": dFAD_leadtime/3600 })
            dssave = pd.concat([dssave, dstemp])

    dssave.to_csv(rf"output\Forecast{[monthindex]}.csv")


if __name__ == "__main__":  

    """Method of running the model on given number of threads, one model runs on each thread sectioned by the monthrange above"""
    import multiprocessing as mp
    import sys
    import itertools

    config_name = sys.argv[1]
    with open(config_name, 'rb') as f:
        config = tomllib.load(f)
    
    totalstartdate = config['startdate']
    totalenddate = config['enddate']
    monthrange = pd.date_range(totalstartdate, totalenddate, freq="MS")
    print()
    # Build tuples of (start, end, index)
    inputs = list(zip(
        monthrange[:-1],
        monthrange[1:],
        range(len(monthrange)-1),
        itertools.repeat(config_name, len(monthrange)-1),
    ))
    print(config_name)
    print(config['depth'])
    #print([filename]*(len(monthrange)-1))
    with mp.Pool(processes=config['parallel_cores']) as pool:
        results = pool.starmap(Run_model, inputs)

    print("Results:", results)

