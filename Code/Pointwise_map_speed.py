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
from functions.funcs import Add_interp_currents

## loaddatasets to be intermpolated 
data_path = './Data/'

dFADs_file = 'SAT_MI_FAD_cleanedspeeds_2026-01-01.parquet'

cmems_file = 'cmems_2022_2026.nc'
oscar_file = 'OSCAR_combined_2021_2026.nc'
era_file =   'ERA5_10m_winds_2021_2026.nc'

output_name = dFADs_file.split('.')[0] + '_mapped_all.parquet'

##Load data
dFADs = gpd.read_parquet(data_path+dFADs_file)

cmems = xr.open_dataset(data_path + cmems_file)
oscar = xr.open_dataset(data_path + oscar_file)
era   = xr.open_dataset(data_path + era_file)
fields = [cmems,oscar,era]

##remove dFAD that are outside of the temportal range of the velocities fields 
maxs = [i.time.values.max() for i in fields]
mins = [i.time.values.min() for i in fields]

dFADs['MaxOfDate'] = pd.to_datetime(dFADs['MaxOfDate'])
dFADs['MinOfDate'] = pd.to_datetime(dFADs['MinOfDate'])
dFADs = dFADs[dFADs["MinOfDate"] > min(mins)]
dFADs = dFADs[dFADs["MaxOfDate"] < max(maxs)]
dFADs = dFADs.reset_index(drop = True)

## Interpolate ield onto dFAD locations 
dFADs = Add_interp_currents(dFADs, cmems.vo, cmems.uo, depth = 13.4671)
dFADs = Add_interp_currents(dFADs, oscar.vo, oscar.uo, tag = '_oscar')
dFADs = Add_interp_currents(dFADs, era.vo, era.uo, tag = '_winds')

dFADs.to_parquet(data_path+output_name)