import pandas as pd
import geopandas as gpd 
import functions.funcs as fad
import pandas as pd
import numpy as np
import shapely as sp
import xarray as xr



class Dataloader():
    def __init__(self, ds: gpd.GeoDataFrame):
        self.ds = ds
        self.dFADs = pd.DataFrame(columns = ["BuoyName","lat", "lon", "TimeStamp", "x_speed", "y_speed"] )

    def First_possitions(self, date:pd.Timestamp, window_length = pd.Timedelta(days =1)):
        """Gets the first possitions of all dFADs on/after the given date
        date: The time at which to get the closed dFADs posstions 
        window_length: the window after the date to load the dFADs 
        """
        endofday = date + window_length
        ds_active = fad.querry_date(self.ds, date) ## All of the active dFADs at this time  #863 total points 
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
            
        ds_locations = ds_locations.query(f"TimeStamp > @date")
        ds_locations = ds_locations.query(f"TimeStamp < @endofday")

        ## remove duplicates
        ds_locations = ds_locations.drop_duplicates(subset=["lat"], keep="first")
        ds_locations = ds_locations.drop_duplicates(subset=["lon"], keep="first").reset_index(drop = True)
        ds_locations = ds_locations.sort_values('TimeStamp').groupby("BuoyName").first().reset_index()
        self.dFADs = pd.concat([self.dFADs,ds_locations.reset_index()]) 
    def Firstposstions_multidays(self, startdate:pd.Timestamp, enddate:pd.Timestamp, window_length:pd.Timedelta):
        """Gets the First possitions of all the dFADs on each day in the range. 
        Used when running the dynammical model over a period of time"""
        Monthdaterange = pd.date_range(startdate, enddate, freq= "D")
        for day in Monthdaterange:
            self.First_possitions(date = day)
    

            

class Alligner():
    """Take Parcels model output and alligns it the true data, 
     so it can be compaired to the true points"""
    def __init__(self, ds:gpd.GeoDataFrame):
        self.True_dFAD_data = Dataloader(ds)
        self.dssave = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])

    def Forcast_snippit(self,ds: gpd.GeoDataFrame, dates, startdate, length)-> gpd.GeoDataFrame: 
        """Grad only the snipbit of dFAD trajectory that lines up with forcast window"""
        ds_s = ds.copy()
        print(ds_s)
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
    def allign_data_MultipleDays(self,output: str, startdate:pd.Timestamp, enddate:pd.Timestamp, forcasttime: pd.Timedelta, month: int):
        output = xr.open_zarr(output, decode_timedelta=True)
        dsout = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])
        masklarge =~ np.isnan(output.lat[:,:].values)

        self.True_dFAD_data.Firstposstions_multidays(startdate, enddate, forcasttime) ## loads the inital forecast possitions
        buoy_list = self.True_dFAD_data.dFADs.BuoyName.tolist()
        ds_filtered = self.True_dFAD_data.ds[self.True_dFAD_data.ds["BuoyName"].isin(buoy_list).reset_index(drop = True)]
        ds_filtered = ds_filtered.reset_index(drop = True)
        ds_short_t = self.Forcast_snippit(ds_filtered, 
                                          self.True_dFAD_data.dFADs['TimeStamp'], 
                                          startdate, 
                                          (enddate-startdate +forcasttime)) ## This could have a delta 7 days


        dFADs = self.True_dFAD_data.dFADs #3This loads the data in the same order as the model does
        dFADs_s = dFADs.iloc[output.Buoyindex.values].reset_index(drop = False)
   
        emptydata = 0
        for i, index in enumerate(output.Buoyindex.values): 
            """Take one forcast and matches it with one True Trjactory"""
            print(i)
            id = dFADs_s.BuoyName[i]
            row =  ds_short_t.query("BuoyName == @id").reset_index(drop = True)
            row = row.iloc[0]
            Times= row["TimeStamp"]
            dFAD_times = (Times - startdate).total_seconds() ## convet to seconds since model started 
            mask = masklarge[i,:] 
            forcast_time_start = (output.time[i,:].values[mask])  ## converts to seconds since model has started. 
            forcast_time_start = forcast_time_start/np.timedelta64(1, "s")
            #print(forcast_time_start/3600)
            #print(dFAD_times/3600)
            dFAD_times_s = dFAD_times[dFAD_times > forcast_time_start[0]]  ## filters true dFADs locations to be inrange with dFAD forcasts 
            dFAD_times_s = dFAD_times_s[dFAD_times_s < forcast_time_start[-1]] 

            #print(dFAD_times_s/3600)
            # print(len(dFAD_times_s))
            lats = output.lat[i,:].values[mask]
            lons = output.lon[i,:].values[mask]
            lat_interp = np.interp(dFAD_times_s, forcast_time_start,lats) # interpolates Forcast times onto true dFAD times 
            lon_interp = np.interp(dFAD_times_s, forcast_time_start, lons) 
            lat_interp = np.insert(lat_interp, 0,np.nan) ## add nan at start of forcast for this is where the initial point is. 
            lon_interp = np.insert(lon_interp, 0,np.nan) ## need to add this into the true data 

            if len(dFAD_times_s) == 0: 
                emptydata += 1
                continue
            if row["geometry"] == None:
                continue
            print(forcast_time_start)
            print(dFAD_times)
            print(dFAD_times_s)
            # print(dFAD_times_s)
            ## get index of first true point thats used in the forcast 
            idx_start = np.where(dFAD_times == dFAD_times_s[0])[0][0] ## fails at case where there is no true data 
            idx_end = np.where(dFAD_times == dFAD_times_s[-1])[0][0]
            ### could raise a probelm is idx_end is already the last value... 
            lon_true, lat_true= row["geometry"].xy
            lon_true = lon_true[idx_start-1:idx_end+1]
            lat_true = lat_true[idx_start-1:idx_end+1]
            Times = Times[idx_start-1:idx_end+1]
            leadtimes = (Times - Times[0])
            leadtimes = leadtimes.total_seconds()/3600
            print(leadtimes)

            Buoylist = [id]*len(lat_true) 
            dstemp= pd.DataFrame({"BuoyID": Buoylist, "Time": Times,
                                    "lat_true": lat_true,"lon_true":lon_true, ""
                                    "lat_forcast": lat_interp, "lon_forcast": lon_interp, 
                                    "leadtime": leadtimes })
            self.dssave = pd.concat([self.dssave, dstemp])

            if i == 20:  ## Error already shows up in 
                self.dssave.to_csv(rf"output\Forecast{[month]}.csv")
                break


if __name__ == "__main__" : 
    if False: 
        ds = gpd.read_parquet(r"..\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")

        monthrange = pd.date_range("2024-01-1","2025-01-1", freq= "MS")
        for month in range(len(monthrange)-1):
            engine = Alligner(ds ,month)
            engine.allign_data_MultipleDays(rf"output\TestParticleFile{month}.zarr", monthrange[month], monthrange[month+1], pd.Timedelta(days = 7), month )
    if True: 
        ds = gpd.read_parquet(r"..\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")
        engine = Alligner(ds)
        engine.allign_data_MultipleDays(rf"output\TestParticleFile0.zarr",
                                         pd.to_datetime("2024-1-1"), pd.to_datetime("2024-1-3"), 
                                         pd.Timedelta(days=7), 0)
        