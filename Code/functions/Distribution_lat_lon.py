import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import geopandas as gpd
from funcs import *

file = r"C:\FATE\Data\Latlondata.csv"

data = pd.read_csv(file, header= None)


timedata = gpd.read_file(r"C:\FATE\Palmyra Data\MI_and_SAT_FAD_positions")
period = timedata['MinOfTimes'].max() - timedata['MinOfTimes'].min() ## period in days
print(period/365)
data = data.set_axis(["Longitude", "Latitude"], axis =1)
lon = data['Longitude']
lat = data['Latitude']


print(lat.max(),lat.min())
print(lon.max(),lon.min())
print(haversine_dist(lat.max(),lat.min(), -161.5, -161.5  ))
##3298.2519105322535

if True:

    total_num = data.shape[0]


    gridded, lonedges, latedges  = np.histogram2d(lon,lat,bins = 100)
    gridded = gridded.T
    gridded = gridded/3
    ##gridded = gridded/total_num 

    lon_avg = np.mean(gridded, axis= 0) ## provided average lon
    lat_avg = np.mean(gridded, axis= 1) ##provides averaged lat 

    max_lon = np.argmax(lon_avg)
    max_lat = np.argmax(lat_avg)
    print(max_lat, max_lon)
    lat_avg[max_lat] = np.mean(lat_avg)
    lon_avg[max_lon] = np.mean(lon_avg)

    lon_coefficents = np.polyfit(lonedges[:-1], lon_avg, 1)
    lat_coefficents = np.polyfit(latedges[:-1], lat_avg, 1)

    bestfit_lon = lon_coefficents[0]*lonedges[:-1] +lon_coefficents[1]
    bestfit_lat = lat_coefficents[0]*latedges[:-1] +lat_coefficents[1]



if True: ## plots it as a scatter plot 
    fig, ax = plt.subplots(1,2, figsize = (10,6))
    ax[0].scatter(lonedges[:-1], lon_avg)
    ax[0].plot(lonedges[:-1],bestfit_lon,label = f"Slope{lon_coefficents[0]:.2f}", color = "k")
    ax[0].set_xlabel("Longitude")
    ax[1].scatter(latedges[:-1], lat_avg)
    ax[1].plot(latedges[:-1], bestfit_lat, label = f"Slope{lat_coefficents[0]:.2f}", color = "k")
    ax[1].set_xlabel("Latitude")
    ax[1].set_ylabel("Frequency")
    ax[0].set_ylabel("Frequency")
    ax[0].legend()
    ax[1].legend()
    fig.suptitle("GPS fixes averaged over Lat and Lon")
    fig.tight_layout()
    fig.savefig(r"C:\FATE\Figures\Distribution_Lat_Lon_Avg.png")




if False: ## Make bar plots 
    ax[0].plot(lonedges[:-1], lon_avg)
    ax[0].set_xlabel("Longitude")
    ax[1].plot(latedges[:-1], lat_avg)
    ax[1].set_xlabel("Latitude")
    fig.suptitle("GPS fixes averaged over Lat and Lon")
    fig.savefig(r"C:\FATE\Figures\Distribution_Lat_Lon_Avg.png")