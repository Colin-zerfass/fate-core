"""Includes two classes: 
to be ran after running the dynamical model. goes from .zarr to .csv files with forecasts alligned to true data
A) "Dataloader": loads dFAD data to be used in Ocean Parcels 
B) "Alligner" : Alligns forecast data with the True dFAD data 

ToRun: 
python Dataloader.py <startdate> <enddate>
"""


import pandas as pd
import geopandas as gpd 
import functions.funcs as fad
import pandas as pd
import numpy as np
import shapely as sp
import xarray as xr



class Dataloader():
    """ Loads posstions of dFADs on a given day. 
    ds: gpd.GeoDataFrame is the master dFAD file 
    dFADs: is the output of the possitions of dFADs on a given day

    Use Case: Loads data and gets possitions for testing hindcasts
    """
    def __init__(self, ds: gpd.GeoDataFrame):
        self.ds = ds
        self.dFADs = pd.DataFrame(columns = ["BuoyName","lat", "lon", "TimeStamp", "x_speed", "y_speed"] )

    def First_possitions(self, date:pd.Timestamp, window_length = pd.Timedelta(days =1),persistencewindow = 2 ):
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
        ## adding speeds to be used in persistence model 
        ds_locations["x_speed_prev"] = (ds_locations.groupby("BuoyName")["x_speed"]
                                        .transform(lambda x: x.rolling(window=persistencewindow, min_periods=1)
                                                   .mean())) ## Calcuates persistence based pervious window
        ds_locations["y_speed_prev"] = (ds_locations.groupby("BuoyName")["y_speed"]
                                        .transform(lambda x: x.rolling(window=persistencewindow, min_periods=1)
                                                   .mean()))
            
        ds_locations = ds_locations.query(f"TimeStamp > @date")
        ds_locations = ds_locations.query(f"TimeStamp < @endofday")

        ## remove duplicates
        ds_locations = ds_locations.drop_duplicates(subset=["lat"], keep="first")
        ds_locations = ds_locations.drop_duplicates(subset=["lon"], keep="first").reset_index(drop = True)
        ds_locations = ds_locations.sort_values('TimeStamp').groupby("BuoyName").first().reset_index()
        self.dFADs = pd.concat([self.dFADs,ds_locations.reset_index()]) 
    def Firstposstions_multidays(self, startdate:pd.Timestamp, enddate:pd.Timestamp, 
                                 window_length:pd.Timedelta, persistencewindow = 2):
        """Gets the First possitions of all the dFADs on each day in the range. 
        Used when running the dynammical model over a period of time"""
        Monthdaterange = pd.date_range(startdate, enddate, freq= "D")
        for day in Monthdaterange[:-1]:
            self.First_possitions(date=day, window_length=window_length,
                                  persistencewindow = persistencewindow )
    

            
class Alligner_test():
    """Take Parcels model output and alligns it the true data, 
     so it can be compaired to the true points"""
    def __init__(self, ds:gpd.GeoDataFrame):
        self.True_dFAD_data = Dataloader(ds)
        self.dssave = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])

    def Forcast_snippit(self,ds: gpd.GeoDataFrame, dates, startdate, length)-> gpd.GeoDataFrame: 
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
    def allign_data_MultipleDays(self,output: str, startdate:pd.Timestamp, enddate:pd.Timestamp, forcasttime: pd.Timedelta, month: int):
        output = xr.open_zarr(output, decode_timedelta=True)

                # Build reverse mapping: integer stored in zarr -> BuoyName.
        all_buoys = sorted(self.True_dFAD_data.ds["BuoyName"].unique())
        int_to_buoy = {idx: name for idx, name in enumerate(all_buoys)}
        buoy_int_ids = output.Buoyindex.values  # 1-D array of encoded BuoyName ints

        # Derive the set of relevant buoys from the zarr output itself so we
        # don't depend on reconstructing the exact same dFADs DataFrame.
        buoy_list = list({int_to_buoy[int(v)] for v in buoy_int_ids})
        ds_filtered = self.True_dFAD_data.ds[self.True_dFAD_data.ds["BuoyName"].isin(buoy_list)].reset_index(drop=True)
        ds_short_t = self.Forcast_snippit(ds_filtered,
                          None,
                          startdate,
                          (enddate-startdate +forcasttime))



        # Load full xarray arrays into numpy once — avoids repeated per-row xarray reads inside the loop
        all_lats = output.lat.values       # shape (n_buoys, n_times)
        all_lons = output.lon.values
        all_times = output.time.values
        buoy_indices = output.Buoyindex.values
        masklarge = ~np.isnan(all_lats)

        print(f"loading Data in month {month}")

        # Pre-build dict lookup to avoid O(n) .query() per iteration
        ds_short_t_by_name = {row["BuoyName"]: row for _, row in ds_short_t.iterrows()}
        # Pre-extract buoy names using the lookup table — avoids needing dFADs DataFrame
        buoy_names = [int_to_buoy[int(v)] for v in buoy_indices]
        startdate_np = np.datetime64(startdate)

        emptydata = 0
        lat_interp_l = []
        lon_interp_l = []
        BuoyID_l = []
        Time_l = []
        lat_true_l = []
        lon_true_l = []
        leadtimes_l = []
        for i in range(len(buoy_indices)): 
            """Take one forcast and matches it with one True Trjactory"""
            if i%100 == 0: 
                print(fr"{i}\{len(buoy_indices)} in month {month}")
            id = buoy_names[i]
            row = ds_short_t_by_name.get(id)
            if row is None:
                continue
            Times= row["TimeStamp"]
            dFAD_times = (Times - startdate).total_seconds() ## convet to seconds since model started 
            mask = masklarge[i,:] 
            ## added for updated times 
            forcast_time_start = all_times[i, mask]
            forcast_time_start = (forcast_time_start - startdate_np) / np.timedelta64(1, "ns") / 1e9

            dFAD_times_s = dFAD_times[(dFAD_times > forcast_time_start[0]) & (dFAD_times < forcast_time_start[-1])]

            lats = all_lats[i, mask]
            lons = all_lons[i, mask]
            lat_interp = np.interp(dFAD_times_s, forcast_time_start,lats) # interpolates Forcast times onto true dFAD times 
            lon_interp = np.interp(dFAD_times_s, forcast_time_start, lons) 
            lat_interp = np.insert(lat_interp, 0,np.nan) ## add nan at start of forcast for this is where the initial point is. 
            lon_interp = np.insert(lon_interp, 0,np.nan) ## need to add this into the true data 

            if len(dFAD_times_s) == 0:
                print(f"data is empty {month}") 
                emptydata += 1
                continue
            if row["geometry"] is None:
                continue

            ## get index of first true point thats used in the forcast
            # np.searchsorted is faster than np.where for sorted arrays
            idx_start = np.searchsorted(dFAD_times, dFAD_times_s[0])
            idx_end = np.searchsorted(dFAD_times, dFAD_times_s[-1])
            lon_true, lat_true= row["geometry"].xy
            lon_true = lon_true[idx_start-1:idx_end+1]
            lat_true = lat_true[idx_start-1:idx_end+1]
            Times = Times[idx_start-1:idx_end+1]
            leadtimes = (Times - Times[0]).total_seconds()/3600

            Buoylist = [id]*len(lat_true) 
            BuoyID_l.extend(Buoylist)
            Time_l.extend(Times)
            lat_true_l.extend(lat_true)
            lon_true_l.extend(lon_true)
            lat_interp_l.extend(lat_interp)
            lon_interp_l.extend(lon_interp)
            leadtimes_l.extend(leadtimes)

        self.dssave =  pd.DataFrame({"BuoyID": BuoyID_l,"Time": Time_l, "lat_true": lat_true_l, "lon_true": lon_true_l, "lat_forcast":lat_interp_l,
                                      "lon_forcast":lon_interp_l, "leadtime":leadtimes_l})
        print(f"{month} has empty data: {emptydata}")
        self.dssave.to_csv(rf"output\Forecast{[month]}.csv")