import geopandas as gpd 
from functions.funcs import generate_longlist

dFADs_data  = 'SAT_MI_FAD_cleanedspeeds_2026-01-01_mapped_all.parquet'

dFAD =gpd.read_parquet(r'Data/' + dFADs_data)

longlist = generate_longlist(dFAD, ['mapped_u', 'mapped_v', 'mapped_u_oscar', 'mapped_v_oscar'])
longlist = longlist.rename(columns = {'mapped_u' : 'mapped_u_glorys' , 'mapped_v': 'mapped_v_glorys'})
longlist.to_csv('Data\SAT_MI_FAD_cleanedspeeds_2026-01-01_mapped_all.csv')