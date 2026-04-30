"""
Generates a point wise comparison of a gridded velocity field to the dFADs possitions 


creates new columns in the dFAD data set of u_mapped and v_mapped (+tag)
eg: u_mapped_OSCAR and v_mapped_OSCAR


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

## loaddatasets to be intermpolated 
data_path = './Data/'
dFADs_file = 'SAT_MI_FAD_cleanedspeeds_2026-01-01.parquet'

cmems_file = 'cmems_2021_2026.nc'
oscar_file = 'OSCAR_combined_2021_2026.nc'
era_file =   'ERA5_10m_winds_2021_2026.nc' 
GEOS_file = 'GEOS_2023.nc'

output_name = dFADs_file.split('.')[0] + '_geos_mapped.parquet'

##Load data
dFADs = gpd.read_parquet(data_path+dFADs_file)

cmems = xr.open_dataset(data_path + cmems_file)
oscar = xr.open_dataset(data_path + oscar_file)
era   = xr.open_dataset(data_path + era_file)

geos = xr.open_dataset(data_path + GEOS_file) ## need to remove to get the entire dFAD dataset back, GEOS is only one year

fields = [cmems,oscar,era, geos]
##remove dFAD that are outside of the temportal range of the velocities fields 
maxs = [i.time.values.max() for i in fields]
mins = [i.time.values.min() for i in fields]

dFADs['MaxOfDate'] = pd.to_datetime(dFADs['MaxOfDate'])
dFADs['MinOfDate'] = pd.to_datetime(dFADs['MinOfDate'])
dFADs = dFADs[dFADs["MinOfDate"] > max(mins)]
dFADs = dFADs[dFADs["MaxOfDate"] < min(maxs)]
dFADs = dFADs.reset_index(drop = True)

## Interpolate ield onto dFAD locations 
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, depth = 13.4671)
print('cmems Done')
dFADs = Add_interp_currents(dFADs, oscar.vo, oscar.uo, tag = '_oscar')
print('OSCAR Done')
dFADs = Add_interp_currents(dFADs, era.vo, era.uo, tag = '_winds')
print('ERA5 Done')
dFADs = Add_interp_currents(dFADs, geos.vo, geos.uo, tag = '_geos')
print('GEOS Done')

# ## interpolates at a range of depths
# dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo,tag = '_1',depth = 0.494)
# print('0.5 m  Done')
# dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, tag = '_5', depth = 5.0782)
# print('5 m Done')
# dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, tag = '_30', depth = 29.4447)
# print('30 Done')


dFADs.to_parquet(data_path+output_name)

dFADs = gpd.read_parquet(data_path + output_name)

## checking amount of nans
from functions.funcs import Column_to_List, list_of_latlon
longlist = pd.DataFrame()
# longlist['mapped_u'] = Column_to_List(dFADs, 'mapped_u')
# longlist['mapped_v'] = Column_to_List(dFADs, 'mapped_v')
# longlist['mapped_u_oscar'] = Column_to_List(dFADs, 'mapped_u_oscar')
# longlist['mapped_v_oscar'] = Column_to_List(dFADs, 'mapped_v_oscar')
# longlist['mapped_u_winds'] = Column_to_List(dFADs, 'mapped_u_winds')
# longlist['mapped_v_winds'] = Column_to_List(dFADs, 'mapped_v_winds')
longlist['mapped_u_geos'] = Column_to_List(dFADs, 'mapped_u_geos')
longlist['mapped_v_geos'] = Column_to_List(dFADs, 'mapped_v_geos')
longlist['TimeStamp'] = Column_to_List(dFADs, 'TimeStamp')
longlist['TimeStamp'] = pd.to_datetime(longlist['TimeStamp'])
longlist['lat'] , longlist['lon'] = list_of_latlon(dFADs, False)
print('Amount of nans:\n')
print(longlist.isna().sum())