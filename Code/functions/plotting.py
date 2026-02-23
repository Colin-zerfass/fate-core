import geopandas as gpd
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import shapely as sp
import xarray as xr
import functions.funcs

def RandTrajectories(ds, amount):
    """plots random Trajectories from dataset provided"""
    length = ds.shape[0]
    ds = ds.reset_index()
    randarray = np.random.randint(0,length,amount)
    fig, ax = plt.subplots()
    for n in range(0,amount):
        line = ds.at[randarray[n],'geometry']
        x,y = line.xy
        ax.plot(x,y)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("latitude")
    return fig, ax

def OneTrajectory(ds,index, ax, window:int = None , itime:int =None, **kwargs):
    """Plots the Trajectory at index, returns the ax """
    line = ds.at[index,'geometry']
    if line is None: 
        return ax
    x,y = line.xy
    if window !=None: ## adds padding to end of the array for sliding window
        x= np.array(x)
        y=np.array(y)
        nans = np.fill(window,np.nan)
        x = np.concatenate((x,nans))
        nans = np.fill(window,np.nan)
        y = np.concatenate((y,nans))
    ax.plot(x,y, **kwargs)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("latitude")
    return ax

def Plotting(ds, amount):
    """Plots the first n "amount" from the dataset"""
    fig, ax = plt.subplots()
    for p in range(0,amount):
        line = ds.at[p,'geometry']
        if line == None: 
            continue
        x,y = line.xy
        name = ds.at[p,'BuoyName']
        ax.plot(x,y,label = name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("latitude")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.scatter(-162.078333, 5.883611,  s = 20,color = "r", label = "Palymra", marker = "*")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig , ax

def AllTrajectories(ds,amount,OutputPath, id = bool):
    """Plots all data in the data set with the "amount" being how many trajectories on each plot, Saves them to specifies output folder"""
    length = ds.shape[0]
    ds = ds.reset_index()
    its = length//amount
    rem = length%amount
    for n in range(0,its): ## amount of full plots to be made 
        i = n*amount
        data  = ds[i:i+amount]
        data = data.reset_index()
        fig, ax = Plotting(data,amount=amount)
        Palmyra_plot(ax)
        fig.savefig(OutputPath+f"_{n}.png")

    if rem > 0: 
        data = ds[-rem:]
        data = data.reset_index()
        fig, ax = Plotting(data,amount=rem)
        fig.savefig(OutputPath+f"_extra.png")

def Palmyra_obj():
    """Returns cords of Palymra as a point"""
    import shapely as shp
    return shp.points(-162.078333, 5.883611,)

def Kingman_obj():
    """Returns cords of Palymra as a point"""
    import shapely as shp
    return shp.points(-162.41667, 6.3833,)

def Palmyra_plot(ax):
    """"Plots Palmyra onto the graph Returns as """
    Palmyra = Palmyra_obj()
    ax.scatter(Palmyra.x,Palmyra.y, marker  = "o", color = "darkgreen", label = "Palmyra", s= 10)
    return ax

def Kingmon_plt(ax):
    kingman = Kingman_obj()
    ax.scatter(kingman.x,kingman.y, marker  = "o", color = "darkgreen", label = "Kingman Reef", s =10)
    return ax

def plotting_direction(lat, lon, delx_list, dely_list,ax,scale = int,  bins= int):
    from scipy.stats import binned_statistic_2d
    x_values = binned_statistic_2d(lon, lat, delx_list, statistic= "mean", bins = bins)
    y_values = binned_statistic_2d(lon, lat, dely_list, statistic= "mean", bins = bins)
    xbound = x_values[1]
    ybound = x_values[2]
    X, Y = np.meshgrid(xbound[:-1], ybound[:-1])
    ax.quiver(X, Y,x_values[0],y_values[0], scale = scale)
    #ax.set_title("Average Direction of dFADS travel")
    ax.set_xlabel("longitude")
    ax.set_ylabel("Latitude")
    return ax


def Add_bathymetry(fig,ax, filepath = r"Data\bath.nc", colorbar = True):
    """Adds Bathymetry to the given ax, provide plt fig and ax, 
    Filepath: Location of Bathymetry,"""
    from matplotlib import cm
    bath = xr.open_dataset(filepath)
    bath_cmap = cm.get_cmap("Blues_r").copy()
    bath_cmap.set_over('green')
    negative_levels = np.linspace(-10000, 0, 11)
    cbr = ax.contourf( bath["lon"], bath["lat"], bath["elevation"], 
                    linestyle = "-", cmap = bath_cmap, alpha = 0.8, levels = negative_levels, extend = "max")
    if colorbar == True:
        fig.colorbar(cbr)
    cbr.set_label("m/s")
    return fig, ax

def NWR_exteriors(data):
    """Returns Exteriors and Interiors from NWP dataset."""
    from shapely.ops import unary_union
    geo = data["geometry"][0]
    geomentry = []

    for polygon in geo.geoms:
        exterior = polygon.exterior
        geomentry.append(sp.Polygon(exterior))
        interior_holes = []
        for interior in polygon.interiors:
            interior_holes.append(sp.Polygon(interior))
        combined_holes = unary_union(interior_holes)
        geomentry.append(combined_holes)
        #multipolygon = sp.MultiPolygon(exteriors)
    labels = ["Palmyra NWR", np.nan,"Kingman NWR", np.nan]
    gpddata = gpd.GeoDataFrame({"labels":labels, "geometry": geomentry})
    return gpddata

def plot_NWPs(ax,data):
    """Plots Palymra and Kingmon Reef, Pass in Shape file as data
    Returns Ax"""
    NWR_ext = NWR_exteriors(data)
    NWR_ext.plot(ax= ax, edgecolor= "darkgreen",alpha = 0.35, column= "labels", legend= True, categorical=True)
    return ax 

def plot_Forcasts(BuoyID:str, dFAD_data, dsforcast: list,startday: int, labels: list, fig, ax, forcastlength = pd.Timedelta(days= 8), pastTrajectory = pd.Timedelta(days = 3)): ## could add getting startime and just get nearest point from that.
    ## getting true data
    perplot = dsforcast[0].query("BuoyID == @BuoyID ") ##SLX+487116 #10 had loops ##16 weird points ##119 ##hit palyra SLX+463917
    truedata = True_dFAD_data(dFAD_data, BuoyID)
    forcasttimes = perplot.query("leadtime == 0 ").reset_index(drop= True)
    starttime = forcasttimes.at[startday, "Time"]
    starttime = pd.to_datetime(starttime)
    starttime = starttime.round('h')
    truedata = truedata[truedata.DateTime < (starttime + forcastlength)]
    truedata = truedata[truedata.DateTime > (starttime - pastTrajectory)]
    ##set x and y lims 
    #fig, ax = plt.subplots(figsize = (6,6))
    ax.plot(truedata.lon_true, truedata.lat_true, label = "dFAD Trajectory", lw= 1.5, color = "k")
    colors = ["limegreen", "orange", "firebrick", "orange","limegreen"]
    ##Get forcast from that starttime 
    for i,ds in enumerate(dsforcast):
        ds = ds.query(f"BuoyID == @BuoyID")
        ds = ds.query(f"initial_time == @starttime").reset_index(drop = True)
        starty =  ds.at[0,"lat_true"]
        startx =  ds.at[0,"lon_true"]
        ax.plot(ds.lon_forcast, ds.lat_forcast, label= labels[i], alpha = 0.75, color = colors[i] )
    deg = 0.5
    #ax.set_ylim([starty -deg, starty +deg+0.2])
    #ax.set_xlim([startx -deg, startx+deg])
    ax.plot(startx, starty, color = "k", lw = 10, alpha= 1)
    ax.set_title(f"Forecasts of dFAD collision \n{BuoyID} \n{starttime}")
    return fig, ax
