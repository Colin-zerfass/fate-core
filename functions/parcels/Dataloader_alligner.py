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
        self._dFADs_list = []

    def First_possitions(self, date:pd.Timestamp, window_length = pd.Timedelta(days =1),persistencewindow = 2 ):
        """Gets the first possitions of all dFADs on/after the given date
        date: The time at which to get the closed dFADs posstions 
        window_length: the window after the date to load the dFADs 
        """
        endofday = date + window_length
        ds_active = fad.querry_date(self.ds, date) ## All of the active dFADs at this time  #863 total points 
        ds_active = ds_active.reset_index(drop = True)
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
        self._dFADs_list.append(ds_locations.reset_index())
        self.dFADs = pd.concat(self._dFADs_list, ignore_index=True)
    def Firstposstions_multidays(self, startdate:pd.Timestamp, enddate:pd.Timestamp, 
                                 window_length:pd.Timedelta, persistencewindow = 2):
        """Gets the First possitions of all the dFADs on each day in the range. 
        Used when running the dynammical model over a period of time"""
        Monthdaterange = pd.date_range(startdate, enddate, freq= "D")
        for day in Monthdaterange:
            self.First_possitions(date=day, window_length=window_length,
                                  persistencewindow = persistencewindow )
    

            
class Alligner():
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
    def allign_data_MultipleDays(self,output: xr.Dataset, startdate:pd.Timestamp, enddate:pd.Timestamp, forcasttime: pd.Timedelta, month: int):


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

        print(f"allining Data in slice {month}")

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
            # if i%100 == 0: 
            #     print(fr"{i}\{len(buoy_indices)} in month {month}")
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

            if len(dFAD_times_s) == 0: 
                emptydata += 1
                continue
            if row["geometry"] is None:
                continue

            ## get index of first true point thats used in the forcast
            # np.searchsorted is faster than np.where for sorted arrays
            idx_start = np.searchsorted(dFAD_times, dFAD_times_s[0])
            idx_end = np.searchsorted(dFAD_times, dFAD_times_s[-1])

            # Guard against idx_start == 0: using idx_start-1 when idx_start is 0
            # causes Python negative indexing to wrap to the last element of the array,
            # producing a wrong-length slice and mismatching lat_interp_l / lat_true_l.
            # When there is no prior true observation (idx_start == 0), skip the NaN
            # "initial point" marker so both lists stay the same length.
            if idx_start > 0:
                lat_interp = np.insert(lat_interp, 0, np.nan)
                lon_interp = np.insert(lon_interp, 0, np.nan)
                slice_start = idx_start - 1
            else:
                slice_start = 0

            lon_true, lat_true= row["geometry"].xy
            lon_true = lon_true[slice_start:idx_end+1]
            lat_true = lat_true[slice_start:idx_end+1]
            Times = Times[slice_start:idx_end+1]
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
        # print(f"{month} has empty data: {emptydata}/{len(buoy_indices)}")
        # self.dssave.to_csv(rf"output\Forecast{[month]}.csv")


_DEPTH_COMMENT         = '# Options are: [ 0.494   1.5414  2.6457  3.8195  5.0782  6.4406  7.9296  9.573  11.405 13.4671 15.8101 18.4956 21.5988 25.2114 29.4447, 55.7643, 109.7293]'
_TAU_COMMENT           = '# Days'
_CURRENTS_FILE_COMMENT = '#options [cmems, OSCAR]'
_FORECAST_LEN_COMMENT  = '#length of Forecasts in Days. ##only changes runtime of model.'

def _write_toml_with_depth_comment(configfile: str, config: dict) -> None:
    """Serialise config with tomli_w then re-attach the depth options comment."""
    import re
    import tomli_w

    with open(configfile, 'wb') as f:
        tomli_w.dump(config, f)

    with open(configfile, 'r', encoding='utf-8') as f:
        text = f.read()

    # Re-attach inline comments stripped by tomli_w
    text = re.sub(r'(^depth\s*=\s*[^\n]*)',          rf'\1  {_DEPTH_COMMENT}',         text, flags=re.MULTILINE)
    text = re.sub(r'(^Tau\s*=\s*[^\n]*)',             rf'\1  {_TAU_COMMENT}',            text, flags=re.MULTILINE)
    text = re.sub(r'(^currents_file\s*=\s*[^\n]*)',   rf'\1  {_CURRENTS_FILE_COMMENT}',  text, flags=re.MULTILINE)
    text = re.sub(r'(^forecast_length\s*=\s*[^\n]*)', rf'\1  {_FORECAST_LEN_COMMENT}',   text, flags=re.MULTILINE)

    with open(configfile, 'w', encoding='utf-8') as f:
        f.write(text)


def compute_bias_corrections(configfile: str) -> None:
    """Compute bias-correction coefficients from collocated dFAD observations
    and write them back into the [GLORYs_correction] table of the TOML config.

    Cases handled
    -------------
    bias=True,   wind=False, stokes=False  ->  Z-scaling of GLORYS at given depth
    bias=True,   wind=True,  stokes=False  ->  2-predictor regression (GLORYS + ERA5 wind)
    stokes=True, bias=False                ->  no coefficients needed; returns early
    stokes=True, bias=True,  wind=False    ->  Z-scaling of (GLORYS + Stokes)
    stokes=True, bias=True,  wind=True     ->  not implemented; raises NotImplementedError
    """
    import numpy as np
    import geopandas as gpd
    import tomli as tomllib
    import functions.settings as settings
    import functions.funcs as func
    from functions.corrections import Calc_Z, Regression

    with open(configfile, 'rb') as f:
        config = tomllib.load(f)

    bias      = config.get('bias', False)
    wind      = config.get('wind', False)
    stokes    = config.get('stokes_drift', False)
    depth_val = config['depth']

    if not bias:
        print("compute_bias_corrections: bias=False — nothing to compute.")
        return

    if stokes and wind:
        raise NotImplementedError("stokes + bias + wind combination is not yet supported.")

    # Map depth value to the nearest pre-mapped parquet column pair
    _depth_cols = {
        0.494:    ('mapped_u_1',   'mapped_v_1'),
        5.0782:   ('mapped_u_5',   'mapped_v_5'),
        29.4447:  ('mapped_u_30',  'mapped_v_30'),
        55.7643:  ('mapped_u_55',  'mapped_v_55'),
        109.7293: ('mapped_u_110', 'mapped_v_110'),
    }
    key_depths = np.array(list(_depth_cols.keys()))
    nearest    = key_depths[np.argmin(np.abs(key_depths - depth_val))]
    u_col, v_col = _depth_cols[nearest]

    # Build longlist from all available dFAD data
    extra = [u_col, v_col, 'mapped_u_winds', 'mapped_v_winds']
    if stokes:
        extra += ['mapped_u_stokes', 'mapped_v_stokes']

    ds       = gpd.read_parquet(settings.dFAD_DATA)
    longlist = func.generate_longlist(ds, extra_columns=extra)

    drop_cols = [u_col, v_col, 'x_speed', 'y_speed', 'mapped_u_winds', 'mapped_v_winds']
    if stokes:
        drop_cols += ['mapped_u_stokes', 'mapped_v_stokes']
    longlist = longlist.dropna(subset=drop_cols).reset_index(drop=True)

    longlist['U']  = longlist['x_speed']       + 1j * longlist['y_speed']
    longlist['W']  = longlist['mapped_u_winds'] + 1j * longlist['mapped_v_winds']
    longlist['Uo'] = longlist[u_col]            + 1j * longlist[v_col]
    if stokes:
        longlist['Uo'] = longlist['Uo'] + (longlist['mapped_u_stokes'] + 1j * longlist['mapped_v_stokes'])

    Uo_mean = complex(longlist['Uo'].mean())
    U_mean  = complex(longlist['U'].mean())
    W_mean  = complex(longlist['W'].mean())

    if not wind:
        Z = Calc_Z(longlist['Uo'], longlist['U'])
        m, n = complex(Z), complex(0)
    else:
        coef = Regression(longlist)
        m, n = complex(coef[0]), complex(coef[1])

    corrections = config.get('GLORYs_correction', {})
    corrections['currents']     = [m.real,       m.imag]
    corrections['wind']         = [n.real,        n.imag]
    corrections['Uo_clim_mean'] = [Uo_mean.real,  Uo_mean.imag]
    corrections['U_dfad_mean']  = [U_mean.real,   U_mean.imag]
    corrections['W_clim_mean']  = [W_mean.real,   W_mean.imag]
    config['GLORYs_correction'] = corrections
    _write_toml_with_depth_comment(configfile, config)

    print(f"compute_bias_corrections: wrote corrections to {configfile}")
    print(f"  depth matched: {nearest:.4f} m  ->  columns {u_col}, {v_col}" + (" + Stokes" if stokes else ""))
    print(f"  mode: {'Z-scaling' if not wind else '2-predictor regression'}")
    print(f"  currents m = {m.real:+.6f} + {m.imag:+.6f}j   |m|={abs(m):.4f} @ {np.angle(m, deg=True):.1f} deg")
    if wind:
        print(f"  wind     n = {n.real:+.6f} + {n.imag:+.6f}j   |n|={abs(n):.4f} @ {np.angle(n, deg=True):.1f} deg")
    print(f"  Uo_clim_mean = [{Uo_mean.real:.6f}, {Uo_mean.imag:.6f}]")
    print(f"  U_dfad_mean  = [{U_mean.real:.6f}, {U_mean.imag:.6f}]")

