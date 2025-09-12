import geopandas as gpd 
import pandas as pd

satlink = r"Palmyra FAD Watch GIS data for NASA (Nov 2024)-selected\Satlink_FAD_positions_sets_070123_063024_PointsToLine.shp"
mi = r"Palmyra FAD Watch GIS data for NASA (Nov 2024)-selected\MI_FAD_positions_sets_062821_063024_PointsToLine\MI_FAD_positions_sets_062821_063024_PointsToLine.shp"

mi_data = gpd.read_file(mi)
sat_data = gpd.read_file(satlink)

rename_dic = {"NAME" : "BuoyName", "Name_ID_ad": "Name_ID", "Length_nm":"Distance_n"}

mi_data.rename(columns=rename_dic, inplace=True) 

print(mi_data.columns)
comb = pd.concat([sat_data, mi_data])
print(comb.columns)
print(comb.shape)

comb.to_file(r"Palmyra FAD Watch GIS data for NASA (Nov 2024)-selected\MI_and_SAT_FAD_positions\MI_and_SAT_FAD_positions.shp")

