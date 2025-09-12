import geopandas as gpd
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from Code.functions.funcs import *
import xarray as xr 
from scipy.interpolate import interpn 

data = gpd.read_file(r"Code\Data\Palmyra Data\MI_and_SAT_FAD_positions")

## Getting time,location, dirction(X,Y,), distance (x,y,xy) and speed (u,v,uv) for each dFAD

if False: ## Adds currents velocity data to a data set for dFADs that samples around 4 hr 
    data = samplefreq(data)
    data = data.query("SampleFreq >3.9").query("SampleFreq <4.1")
    print(data.shape)
    data = data.reset_index()
    data.to_csv(r"Code\Data\speedsat_4hr.csv")
    if True:
        data, delx_list, dely_list  = add_distance_collumns(data)
        data = add_time_collumns(data)
        data = data.reset_index()

        ds = xr.open_dataset(r"Code\Data\cmems(3).nc")
        vo  = ds['vo'] ## this is y velocity
        uo = ds['uo'] ## this is x velocity

        data = interp_currents(data, vo ,uo)
        newdata = pd.DataFrame({"BuoyName" :data["BuoyName"], "Mapped_u": data["mapped_u"], "Mapped_v": data["mapped_v"]})
        newdata.to_pickle(r"Code\Data\Mapped_speeds.pkl")
        #newdata.to_parquet(r"Code\Data\Mapped_speeds.parquet")
        #data.to_file(r"Data\Mapped_data\Mapped_4hr_period.shp")
        data.to_parquet(r"Code\Data\Mapped_4hr_period.parquet")

if False: ##Makes plot of nonunique dFADs
    non_unique = nonUnique_tracks(data)
    one_group = non_unique[-3:]
    print(one_group["MinOfDate"])
    AllTrajectories(one_group, 3,r"Figures\sameID")

if False: ##Makes figure for Direction Vectors 
    data, delx_list, dely_list = add_distance_collumns(data)
    lat, lon = list_of_latlon(data)

    fig, ax = plt.subplots(1,2, figsize = (10,6))
    plotting_direction(lat, lon, delx_list=delx_list, dely_list=dely_list, ax = ax[0], scale =15, bins = 20)
    plotting_direction(lat, lon, delx_list=delx_list, dely_list=dely_list, ax = ax[1], scale = 10, bins = 40)

    plotting_zoom(0.5,ax= ax[1])
    Palmyra_plot(ax[1])
    Palmyra_plot(ax[0])
    fig.suptitle("Average Direction of dFADS travel")
    fig.tight_layout()
    fig.savefig(r"Figures\Direction_plot_correct.png")

if False: ## want to plot diffence between 4hr intervals and 24 hr intervals
    data = samplefreq(data)
    shortperiod = data.query("SampleFreq > 3").query("SampleFreq <4")
    longperiod = data.query("SampleFreq > 3").query("SampleFreq <4")
    RandTrajectories(data,10)

if False: ## Plotting the distance to palmyra 
    data = distance_from_Palymra(data)
    if False: ## make Historgram
        fig, ax = plt.subplots(1,2, figsize = (10,6))
        ax[0].hist(data["distance_km"],bins = 15)
        closest = data.query("distance_km < 50")
        ax[1].hist(closest["distance_km"],bins = 10)
        fig.supxlabel("Distance (km)")
        fig.supylabel("Frequency")
        fig.suptitle("Closest Distance to Palmyra for all dFADS")
        fig.savefig(r"Figures\DistanceToPalmyra_hist.png")

    Collision = data.query("distance_km < 5")
    AllTrajectories(Collision, 1, r"Figures\Collision\within5km", id = True)

if False: ## Plots all the Trajectories sorted by month
    ##lets plot all the Trajectories for all the months
    mondata = monthly_data(data)
    months = ['Jan', "Feb", "Mar", "Apr", "May", "Jun","Jul","Aug", "Sep", "Oct", "Nov", "Dec"]
    for n in range(0,12):
        AllTrajectories(mondata[n],10,rf"Figures\Monthly_plots\{n+1}\{months[n]}")

if False: ## plotting All data from month of Jan
    mondata = monthly_data(data)
    AllTrajectories(mondata[0],10,r"Figures\Monthly_plots\Jan")

if False: ###Plotting Random monthly data
    mondata = monthly_data(data)
    fig , ax = RandTrajectories(mondata[0],10)
    fig.savefig(r"Figures\Jan_plot.png")

