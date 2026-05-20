import geopandas as gpd 
import numpy as np 
import pandas as pd 
import sys 
from pathlib import Path
import functions.settings as settings
import functions.funcs as fad
from functions.parcels.Dataloader_alligner import Dataloader
import shapely as sp

# Create the persistence model
def persistence_model(merged_df, start_time, window_hrs,
                       max_horizon_hrs=None, define_persistence_as='mean', velocity_window=2,
                         dt_freq=None, target_date = pd.Timestamp):
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




def generate_peristence_forecasts(dfADs, startdate:pd.Timestamp, enddate:pd.Timestamp):
    loader = Dataloader(ds)
    print('finding dFADs')
    loader.Firstposstions_multidays(startdate=startdate, enddate=enddate, window_length= pd.Timedelta(1, unit='day'))
    all_dFADs = loader.dFADs.reset_index(drop=True)
    all_dFADs["TimeStamp"] = pd.to_datetime(all_dFADs["TimeStamp"])
    all_dFADs["_date"] = all_dFADs["TimeStamp"].dt.normalize()

    dssave_list = []
    print(f"Total dFADs: {len(all_dFADs)}")

    for i in range(len(all_dFADs)):
        BuoyID = all_dFADs.at[i, "BuoyName"]
        target_time = all_dFADs.at[i, "TimeStamp"]
        target_date = all_dFADs.at[i, "_date"]
        dFAD = ds.query(f"BuoyName == @BuoyID").reset_index(drop=True)
        speed = dFAD.xy_speed.values[0]
        times = dFAD.TimeStamp.values[0]
        lat, lon = fad.list_of_latlon(dFAD, False)
        onedFAD = pd.DataFrame({"DateTime": times, "Latitude_true": lat, "Longitude_true": lon, "speed_ms_true": speed})

        future, time = persistence_model(onedFAD, target_time, 10, 8*24, velocity_window=2, dt_freq=pd.Timedelta(hours=4), target_date=target_date)
        future["BuoyID"] = [BuoyID]*len(future)

        dssave_list.append(future)
        if i%1000 == 0:
            print(f" {i}/{len(all_dFADs)}")
    dssave = pd.concat(dssave_list, ignore_index=True)
    dssave = dssave.drop(columns=['speed_ms_persistence'])
    dssave = dssave.rename(columns={"Latitude_true": "lat_true", "Longitude_true": "lon_true",
                                    "Latitude_persistence": "lat_forcast",
                                    "Longitude_persistence": "lon_forcast",
                                    "lead_time_hours": "leadtime", "DateTime": "Time"})
    dssave.to_csv(settings.FORECAST_DIR / 'persistence.csv')

if __name__ == '__main__':
    ds = gpd.read_parquet(settings.dFAD_DATA)

    startdate = pd.Timestamp("2022-01-01")
    enddate = pd.Timestamp("2025-07-01")
    generate_peristence_forecasts(ds, pd.Timestamp('2022-01-01'), pd.Timestamp('2025-12-31'))

