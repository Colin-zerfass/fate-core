import geopandas as gpd 
import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np
import shapely as shp

data = gpd.read_file(r"C:\FATE\Palmyra Data\MI_and_SAT_FAD_positions")

data["numpoints"] = shp.get_num_points(data["geometry"])
data["SampleFreq"] = data["Diff_days"]/data["numpoints"]*24 ## gives average rate of data in hours

DaylySamples = data.query('SampleFreq < 24')

bins  = np.linspace(0,24,24)

fig, ax = plt.subplots()
ax.hist(DaylySamples["SampleFreq"], bins = bins)
ax.set_xlabel("hours")
ax.set_xticks(bins, minor=True)
ax.set_title("Average Frequancy of GPS locations received per dFAD")
fig.tight_layout()
fig.savefig(r"C:\FATE\Figures\Timedata_histogram.png")