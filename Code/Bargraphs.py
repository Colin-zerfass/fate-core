import geopandas as gpd 
import pandas as pd 
import matplotlib.pyplot as plt
from  datetime import datetime
import numpy as np 

data = gpd.read_file(r"C:\FATE\Palmyra FAD Watch GIS data for NASA (Nov 2024)-selected\MI_and_SAT_FAD_positions")
data.shape ##Checking if the files were combined correctly

fig, ax = plt.subplots()
ax.hist(data['MinOfDate'], bins = 20)
ax.tick_params(axis='x', labelrotation=45)
ax.set_xlabel("start Dates of dFADs ")
ax.set_title("Histogram of start dates")
fig.savefig(r"C:\FATE\Figures\Historgram_entrytime.png")


shorten = data.query('MinOfDate > 2022' )
shorten = shorten.query('MinOfDate < 2024')
print(shorten['MinOfDate'].min())
print(shorten['MinOfDate'].max()) 
##only includes from two years so propery weighted
shorten["Month"] = shorten['MinOfDate'].dt.strftime("%m") ## removes year
print(shorten['Month'].head(5))
shorten['Month'] = shorten['Month'].sort_values()
sorted_counts = shorten['Month'].value_counts().sort_index()

fig, ax = plt.subplots()
line = np.linspace(1,12,12)
ax.bar(line, sorted_counts)
ax.set_xlabel("Months ")
ax.set_title("bargraph of time dFADS enter Area")
fig.savefig(r"C:\FATE\Figures\bargraph_of_Month dFADS.png")