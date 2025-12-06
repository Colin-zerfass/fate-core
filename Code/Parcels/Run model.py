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

from functions.funcs import *

##load some data 
## Loads the Velosity field
fname = r"..\Data\krigging_field_40.nc"
field = xr.open_dataset(fname )
#field = field.rename({"longitude":"lon", "latitude": "lat"}) ## if using cmems

##Loads the dFADs 
ds = gpd.read_parquet(r"..\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")
monthrange = pd.date_range("2024-01-1","2025-01-1", freq= "MS")
for month in range(len(monthrange)-1):
    daterange = pd.date_range(monthrange[month], monthrange[month+1])
    dssave = pd.DataFrame()
    dssave = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])
    for day in daterange:
        target_date = day ## picks dFAD locations one day after this date 
        print(target_date)

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

        ##Filter Timestep by certain threshhold to get locations of FADS within closes  
        ## UPDATE:This might be better to interp these onto the specific time. 
        hourlim = 24
        time_threshhold  = pd.Timedelta(hours= hourlim)
        time_upper  = target_date + time_threshhold ## This is set for dFADs one day after the date 
        time_lower = target_date 
        ds_locations = ds_locations.query(f"TimeStamp > @time_lower")
        ds_locations = ds_locations.query(f"TimeStamp < @time_upper")
        print(f"Amount of sampled dFAD within {hourlim} hrs : {len(ds_locations)}")
        ds_locations = ds_locations.drop_duplicates(subset=["lat"], keep="first")
        ds_locations = ds_locations.drop_duplicates(subset=["lon"], keep="first").reset_index(drop = True)

        ## New get only the first point of the day for the forcast.

        dFADs = ds_locations.sort_values('TimeStamp').groupby("BuoyName").first()
        print(f"Number of Unique dFADs/ points avalable: {len(dFADs)}")
        if len(dFADs) < 1: 
            continue
        dFADs = dFADs.reset_index()

        ## Make the model... 
        filenames = {"uo": fname, "vo": fname}
        variables  = {"U": "uo", "V": "vo"}
        dimensions = {"lat": "lat", "lon": "lon"}
        field_t = field.sel(time = target_date, method = "nearest").drop_vars("time") ## IF CMEMS add depth = 15 argument 
        runtime = pd.Timedelta(days =8)

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
            
        dFADs["timedelta"] = (dFADs.TimeStamp - target_date).dt.total_seconds()

        Particles = parcels.ScipyParticle.add_variable("age", initial = 0) 
        Particles = Particles.add_variable("Buoyindex", to_write = 'once')

        pset = parcels.ParticleSet.from_list(fieldset, pclass = Particles , lon = dFADs.lon.to_list(), 
                                            lat = dFADs.lat.to_list() , time = dFADs.timedelta, Buoyindex = dFADs.index.values) 

        output_memorystore = zarr.storage.MemoryStore()
        output_file = pset.ParticleFile(name = output_memorystore, outputdt =timedelta(hours= 1))


        pset.execute([parcels.AdvectionRK4, boundryCondition, Age], 
                    runtime = timedelta(days = 8), ##this should be 8 days 
                    dt = timedelta(minutes =5), 
                    output_file = output_file, 
                    )


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

        for i, index in enumerate(output.Buoyindex.values): 
            id = dFADs.BuoyName[index]
            row=  ds_short_ts.iloc[i]
            Times= row["TimeStamp"]
            starttime = Times[0]
            dFAD_leadtime = (Times - starttime).astype("int64")/1e9 ## convets leadtime to seconds
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

    dssave.to_csv(rf"output\Forecast{[month]}.csv")

