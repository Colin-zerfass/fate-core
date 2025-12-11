import geopandas as gpd
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
from Code.functions.funcs import *
import xarray as xr 
from scipy.interpolate import interpn 

data = gpd.read_parquet(r"Code\Data\Palmyra Data\SAT_MI_FAD_cleanedspeeds.parquet")

dataNWR = gpd.read_file(r"Code\Data\Palmyra_Shapefiles",  layer = 'PAL_KING_NWR_12nm')

if False: ## Cleans the dFAD file based on speed positions and saves , This is an old method od cleaning check Cleaning_data.py
    ### Method for cleaning data: Read Remove_speeds_high_low for method & check documentation.
    ##This is done before mapping otherwise needs to mask out Mapped velosities.
    ###Adding columns needed to the data
    ###This is an old method od cleaning check Cleaning_data.py
    data = samplefreq(data)
    data, delx_long, dely_long= add_distance_collumns(data)
    data = add_time_collumns(data)
    data = Add_x_y_speed_collums(data)
    data= add_time_collumns(data)

    ## removing High speed points
    dataclean = Remove_speeds_high_low(data)

    ### Removing The bad points in Geometry 
    dataclean["new_geometry"] = dataclean.apply(Filter_geometry_obj, axis = 1)
    dataclean["geometry"] = dataclean["new_geometry"]
    dataclean = dataclean.drop(columns = ["new_geometry"])
    valid_mask = ~dataclean.geometry.isna() & ~dataclean.geometry.is_empty
    dataclean = dataclean[valid_mask]

    ##Mask columns that have been calculated from the orignial geometry item 
    columnlist = ['x_deg', 'y_deg', 'x_km', 'y_km', 'xy_km', 'timelist' ]
    for names in columnlist: 
        print(names)
        dataclean[f"new_{names}"] = dataclean.apply(Filter_Rows, axis =1, column = names)
        dataclean[f"{names}"] = dataclean[f"new_{names}"] 
        dataclean = dataclean.drop(columns = [f"new_{names}"])
    
    print(dataclean.columns)
    print(len(dataclean["x_deg"][0]))
    dataclean.to_parquet(r"Code\Data\Palmyra Data\MI_and_SAT_FAD_Cleaned.parquet")



if False :## Getting time,location, dirction(X,Y,), distance (x,y,xy) and speed (u,v,uv) for each dFAD

    ### Adding Distance collumns
    
    data = data.query("SampleFreq >3.9").query("SampleFreq <4.1")
    data = add_distance_collumns(data)
    data = add_time_collumns(data)
    data = Add_x_y_speed_collums(data)
    data = Add_interp_currents(data)
    data = Add_Mapped_speeds(data)
    data = Add_Delta_speeds(data)
    

    data.to_parquet(r"Code\Data\Mapped_4hr_period.parquet")

if True: ## Adds currents velocity data to a data set for dFADs that samples around 4 hr 
    #data.to_csv(r"Code\Data\speedsat_4hr.csv")
    if True:
        #data = remove_no_TimeStamp(data) ##this is not needed and do not do, Changes the size of arrays
        #data = Add_x_y_speed_collums_TimeStamp(data)
        #data, delx_list, dely_list  = add_distance_collumns(data)
        data["MinOfDate"] = pd.to_datetime(data["MinOfDate"])
        data["MaxOfDate"] = pd.to_datetime(data["MaxOfDate"])
        #data = data.iloc[2920:]
        data = data.reset_index(drop = True)

        ds = xr.open_dataset(r"Code\Data\cmems.nc")
        vo  = ds['vo'] ## this is y velocity
        uo = ds['uo'] ## this is x velocity
        data[data["MinOfDate"] > ds['time'].to_numpy().min()]
        data[data["MaxOfDate"] < ds['time'].to_numpy().max()]
        data = data.reset_index(drop = True)

        data = Add_interp_currents(data, vo ,uo)
        #newdata = pd.DataFrame({"BuoyName" :data["BuoyName"], "Mapped_u": data["mapped_u"], "Mapped_v": data["mapped_v"]})
        #newdata.to_pickle(r"Code\Data\Mapped_speeds.pkl")
        #newdata.to_parquet(r"Code\Data\Mapped_speeds.parquet")
        #data.to_file(r"Data\Mapped_data\Mapped_4hr_period.shp")
        data.to_parquet(r"Code\Data\Mapped_SAT_MI_Cleanedspeeds.parquet")

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
    ax[0] = plot_NWPs(ax[0], dataNWR)
    ax[1] = plot_NWPs(ax[1], dataNWR)
    Palmyra_plot(ax[1])
    Palmyra_plot(ax[0])
    Kingmon_plt(ax[0])
    Kingmon_plt(ax[1])
    fig.suptitle("Average Direction of dFADS travel")
    fig.tight_layout()
    fig.savefig(r"Figures\Direction_plot_correct2.png")

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

