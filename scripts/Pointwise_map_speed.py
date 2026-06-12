"""
Generates a point wise comparison of a gridded velocity field to the dFADs possitions 


creates new columns in the dFAD data set of u_mapped and v_mapped (+tag)
ie: u_mapped_OSCAR and v_mapped_OSCAR


Standard names of the mapped speeds should be: 
cmems: 'u_mapped', 'v_mapped' 
oscar: 'u_mapped_oscar', 'v_mapped_oscar' 
ERA5(wind): 'u_mapped_winds', ' v_mapped_winds'

"""

import geopandas as gpd
import xarray as xr 
import numpy as np 
import pandas as pd
import geopandas as gpd
from functions.funcs import Add_interp_currents
import functions.settings as settings

## loaddatasets to be intermpolated 
data_path = settings.DATA_DIR
dFADs_file = r'SAT_MI_FAD_cleanedspeeds_2026-01-01.parquet'

##Load data
dFADs = gpd.read_parquet(data_path / dFADs_file)

cmems = xr.open_dataset(settings.GLORYS_FILE)
oscar = xr.open_dataset(settings.OSCAR_FILE)
era   = xr.open_dataset(settings.ERA5_FILE)

#geos = xr.open_dataset(data_path + GEOS_file) ## need to remove to get the entire dFAD dataset back, GEOS is only one year

fields = [cmems,oscar,era]
##remove dFAD that are outside of the temportal range of the velocities fields 
maxs = [i.time.values.max() for i in fields]
mins = [i.time.values.min() for i in fields]

dFADs['MaxOfDate'] = pd.to_datetime(dFADs['MaxOfDate'])
dFADs['MinOfDate'] = pd.to_datetime(dFADs['MinOfDate'])
dFADs = dFADs[dFADs["MinOfDate"] > max(mins)]
dFADs = dFADs[dFADs["MaxOfDate"] < min(maxs)]
dFADs = dFADs.reset_index(drop = True)

## Interpolate ield onto dFAD locations 
print('starting GLORYs')
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, depth = 13.4671)
print('GLORYs Done')
print('starting OSCAR')
dFADs = Add_interp_currents(dFADs, oscar.vo, oscar.uo, tag = '_oscar')
print('OSCAR Done')
print('starting ERA5')
dFADs = Add_interp_currents(dFADs, era.vo, era.uo, tag = '_winds')
print('ERA5 Done')
print('Starting Stokes')
dFADs = Add_interp_currents(dFADs, era.vst, era.ust, tag = '_stokes')
print('Stokes Done')

## interpolates at a range of depths
print('starting 0.5m ')
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo,tag = '_1',depth = 0.494)
print('0.5 m  Done')
print('starting 5m')
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, tag = '_5', depth = 5.0782)
print('5 m Done')
print('starting 30m')
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, tag = '_30', depth = 29.4447)
print('30 Done')
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, tag = '_30', depth = 29.4447)
print('30 Done')
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, tag = '_55', depth = 55.7643)
print('50 Done')
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, tag = '_110', depth = 109.7293)
print('100 Done')


dFADs.to_parquet(settings.dFAD_DATA)

dFADs = gpd.read_parquet(settings.dFAD_DATA)

## checking amount of nans
from functions.funcs import Column_to_List, list_of_latlon
longlist = pd.DataFrame()
longlist['mapped_u'] = Column_to_List(dFADs, 'mapped_u')
longlist['mapped_v'] = Column_to_List(dFADs, 'mapped_v')
longlist['mapped_u_oscar'] = Column_to_List(dFADs, 'mapped_u_oscar')
longlist['mapped_v_oscar'] = Column_to_List(dFADs, 'mapped_v_oscar')
longlist['mapped_u_winds'] = Column_to_List(dFADs, 'mapped_u_winds')
longlist['mapped_v_winds'] = Column_to_List(dFADs, 'mapped_v_winds')
longlist['TimeStamp'] = Column_to_List(dFADs, 'TimeStamp')
longlist['TimeStamp'] = pd.to_datetime(longlist['TimeStamp'])
longlist['lat'] , longlist['lon'] = list_of_latlon(dFADs, False)
print('Amount of nans:\n')
print(longlist.isna().sum())