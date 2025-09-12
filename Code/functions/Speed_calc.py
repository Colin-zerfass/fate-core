import geopandas as gpd 
import pandas as pd 
import numpy as np 
import shapely as shp 
from funcs import *


file_path = r"C:\FATE\Palmyra Data\MI_and_SAT_FAD_positions"

data = gpd.read_file(file_path)

data = samplefreq(data)

data4 = data.query("SampleFreq > 3.6").query("SampleFreq < 4.4")
data4 = data4.reset_index()

data4, test, test2 = add_distance_collumns(data4)

# print(data4["x"])
# print(data4["y"])
# print(data4["xy_km"])
data4 = Add_x_y_speed_collums(data4)








