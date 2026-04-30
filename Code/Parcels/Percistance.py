import geopandas as gpd 
import numpy as np 
import pandas as pd 
import sys 
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
from functions.funcs import *


# Create the persistence model
def persistence_model(merged_df, start_time, window_hrs, max_horizon_hrs=None, define_persistence_as='mean', velocity_window=2, dt_freq=None, target_date = pd.Timestamp):
    df = merged_df.copy()

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df=df.sort_values(['DateTime']).reset_index(drop=True)
 
    # Set start time as the start time of true trajectory
    if start_time is None:
        if len(df)>=velocity_window:
            start_time=df['DateTime'].iloc[velocity_window-1]
        
        else:
            start_time=df['DateTime'].iloc[0]
        
    start_time = pd.to_datetime(start_time)
 
    # Base row
    row_t0 = df.loc[(df['DateTime'] ==start_time)]
    #print(row_t0)
    lat_t0 = row_t0["Latitude_true"].values
    lon_t0 = row_t0["Longitude_true"].values
     
    # Persistence base
    if window_hrs <= 0:
        base = row_t0[['Latitude_true', 'Longitude_true', 'speed_ms_true']].iloc[0].to_dict()
    else:
        t_min = start_time - pd.Timedelta(hours=window_hrs)
        past = df[(df['DateTime'] > t_min) & (df['DateTime'] <= start_time)]
        if past.empty:
            past = row_t0
        if define_persistence_as == 'last':
            base_row = past.sort_values('DateTime').iloc[-1]
        elif define_persistence_as == 'median':
            base_row = past.median()
        else:
            base_row = past.mean()
        base = {
            'Latitude_true': base_row['Latitude_true'],
            'Longitude_true': base_row['Longitude_true'],
            'speed_ms_true': base_row['speed_ms_true']
        }
    
    # Velocity
    past_window = df[df['DateTime'] <= start_time].sort_values('DateTime').tail(velocity_window)
    if len(past_window) >= 2:
        dt_seconds = (past_window['DateTime'].iloc[-1] - past_window['DateTime'].iloc[0]).total_seconds()
        if dt_seconds>0:
            v_latitude = (past_window['Latitude_true'].iloc[-1] - past_window['Latitude_true'].iloc[0]) / dt_seconds
            v_longitude = (past_window['Longitude_true'].iloc[-1] - past_window['Longitude_true'].iloc[0]) / dt_seconds
        else:
            v_latitude = 0
            v_longitude = 0
    else:
        v_latitude = 0
        v_longitude = 0
    
    if dt_freq is None:
        dt=df['DateTime'].diff().median()
        if pd.isna(dt):
            dt=pd.Timedelta(hours=1)
        freq=pd.Timedelta(dt).to_pytimedelta()
 
        end_time=start_time+pd.Timedelta(hours=max_horizon_hrs)
        future_times=pd.date_range(start=start_time, end=end_time, freq=dt)
    else:
        end_time=start_time+pd.Timedelta(hours=max_horizon_hrs)
        future_times=pd.date_range(start=start_time, end=end_time, freq=dt_freq)

    end_time=target_date+pd.Timedelta(hours=max_horizon_hrs)
    ### Setting to Future points 
    future_points = df[(df['DateTime'] <= end_time) & (df['DateTime'] >= start_time)]
    future_times = future_points["DateTime"].to_list()
    future_lats = future_points["Latitude_true"].to_list()
    future_lons = future_points["Longitude_true"].to_list()
    future=pd.DataFrame({'DateTime': future_times})
    dt_future=(future['DateTime']-start_time).dt.total_seconds()

    # Extrapolate
    future['Latitude_persistence'] = lat_t0 + v_latitude * dt_future
    future['Longitude_persistence'] = lon_t0 + v_longitude * dt_future
    future['speed_ms_persistence'] = base['speed_ms_true']
    future['lead_time_hours']=dt_future/3600.0
    future['Longitude_true'] = future_lons
    future['Latitude_true'] = future_lats
    return future, pd.to_datetime(start_time)
## Snippit of true Trajectory
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


ds = gpd.read_parquet(r"..\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")
monthrange = pd.date_range("2022-01-01","2025-07-01", freq= "MS")
for month in range(len(monthrange)-1):
    daterange = pd.date_range(monthrange[month], monthrange[month+1])
    dssave = pd.DataFrame()
    #dssave = pd.DataFrame(columns = ["BuoyID","Time", "lat_true", "lon_true", "lat_forcast", "lon_forcast", "leadtime"])
    dssave = pd.DataFrame(columns = [ "BuoyID" , "DateTime", "Latitude_true", "Longitude_true", "Latitude_persistence",  
                                     "Longitude_persistence",  "speed_ms_persistence"  ,"lead_time_hours"])
    for day in daterange:
        target_date = day ## picks dFAD locations on this date 
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
        # if len(dFADs) < 1: 
        #     continue
        dFADs = dFADs.reset_index()


        buoy_list = dFADs.BuoyName.tolist()
        ds_filtered = ds_active[ds_active["BuoyName"].isin(buoy_list)].reset_index(drop = True)
        ds_short_t = Forcast_snippit(ds_filtered, dFADs.TimeStamp, target_date, pd.Timedelta(days = 8))


        for i in range(len(dFADs)):
            ### this has to be for each Bouy
            BuoyID = dFADs.at[i, "BuoyName" ]
            #print(BuoyID)
            target_time = dFADs.at[i, "TimeStamp" ]
            #Time = dFADs.at[2,"TimeStamp"]
            dFAD = ds.query(f"BuoyName == @BuoyID").reset_index(drop = True)
            speed = dFAD.xy_speed.values[0]
            times = dFAD.TimeStamp.values[0]
            lat,lon = list_of_latlon(dFAD, False)
            onedFAD = pd.DataFrame({"DateTime" : times, "Latitude_true": lat, "Longitude_true": lon, "speed_ms_true": speed})

            future, time = persistence_model(onedFAD, target_time, 10,8*24, velocity_window = 2, dt_freq= pd.Timedelta(hours= 4), target_date = target_date)
            future["BuoyID"] = [BuoyID]*len(future) 
    
            dssave = pd.concat([dssave, future])

    dssave = dssave.drop(columns= ["BouyID", 'speed_ms_persistence'])

  # to fix persistance column names intial
    dssave=dssave.rename(columns={"Latitude_true" : "lat_true", "Longitude_true": "lon_true", 
                    "Latitude_persistence": "lat_forcast",
                      "Longitude_persistence": "lon_forcast", 
                      "lead_time_hours": "leadtime" , "DateTime": "Time"})

    dssave.to_csv(rf"output\Percistance_Forecast{[month]}.csv")
