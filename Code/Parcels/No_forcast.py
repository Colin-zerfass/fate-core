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
monthrange = pd.date_range("2023-01-1","2024-01-1", freq= "MS")
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
        dFADs = dFADs.reset_index() ## list of all the initial possitions 


        buoy_list = dFADs.BuoyName.tolist()
        ds_filtered = ds_active[ds_active["BuoyName"].isin(buoy_list)].reset_index(drop = True)

        def get_buoy_row(df, buoy_id, single=True):
            mask = df['BuoyName'] == buoy_id
            if single:
                rows = df.loc[mask]
                if len(rows) == 0:
                    return None
                return rows.iloc[0]
            return df.loc[mask]


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
        
        dsout = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])

        for i, id in enumerate(dFADs.BuoyName): 
            row=  get_buoy_row(ds_short_t, id)
            Times= row["TimeStamp"]
            starttime = Times[0]
            dFAD_leadtime = (Times - starttime).total_seconds()#.astype("int64")/1e9 ## convets leadtime to seconds 
            if row["geometry"] == None:
                continue
            lon_true, lat_true= row["geometry"].xy
            Bouylist = [id]*len(lat_true) 
            lat_forcast = [dFADs.iloc[i].lat]*len(lat_true)
            lon_forcast = [dFADs.iloc[i].lon]*len(lon_true)
            

            dstemp= pd.DataFrame({"BuoyID": Bouylist, "Time": Times,
                                   "lat_true": lat_true,"lon_true":lon_true, ""
                                   "lat_forcast": lat_forcast, "lon_forcast": lon_forcast, ## only need to change these 
                                   "leadtime": dFAD_leadtime/3600 })
            dssave = pd.concat([dssave, dstemp])

    dssave.to_csv(rf"output\Forecast{[month]}.csv")

