import geopandas as gpd
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np
import shapely as sp
import xarray as xr

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

def OneTrajectory(ds,index, ax, window:int = None , itime:int =None):
    line = ds.at[index,'geometry']
    x,y = line.xy
    if window !=None: ## adds padding to end of the array for sliding window
        x= np.array(x)
        y=np.array(y)
        nans = np.fill(window,np.nan)
        x = np.concatenate((x,nans))
        nans = np.fill(window,np.nan)
        y = np.concatenate((y,nans))
    ax.plot(x,y)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("latitude")
    return ax

def Plotting(ds, amount):
    fig, ax = plt.subplots()
    for p in range(0,amount):
        line = ds.at[p,'geometry']
        x,y = line.xy
        name = ds.at[p,'Name_ID']
        ax.plot(x,y,label = name)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("latitude")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.scatter(-162.078333, 5.883611,  s = 20,color = "r", label = "Palymra", marker = "*")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    return fig , ax

def AllTrajectories(ds,amount,OutputPath, id = bool):
    """Plots all data in the data set with the "amount" being how many trajectories on each plot"""
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
        
def monthly_data(data):
    monthly_data = []

    for i in range(1,13):
        month = data.query(f"Mon_min == {i}")
        monthly_data.append(month)
    return monthly_data

def haversine_dist(lat1, lon1, lat2, lon2):
    """Returns distance between two points in Km (always positive)"""
    from math import radians, sin, cos, atan2, sqrt
    # Convert lat and lon to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
 
    dist_lon = lon2_rad-lon1_rad
    dist_lat = lat2_rad-lat1_rad
 
    a = sin(dist_lat/2)**2+cos(lat1_rad)*cos(lat2_rad)*sin(dist_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
 
    R = 6371.0
 
    distance = R*c
    return distance

def haversine_dist_with_negetive(lat1, lon1, lat2, lon2):
    """ONLY to be used when calcuating Distance in X and Y indivigually 
    Returns distance between two points in Km 
    - xdirection and negetive y direction are allowed"""
    from math import radians, sin, cos, atan2, sqrt
    # Convert lat and lon to radians
    lat1_rad, lon1_rad = radians(lat1), radians(lon1)
    lat2_rad, lon2_rad = radians(lat2), radians(lon2)
 
    dist_lon = lon2_rad-lon1_rad
    dist_lat = lat2_rad-lat1_rad
 
    a = sin(dist_lat/2)**2+cos(lat1_rad)*cos(lat2_rad)*sin(dist_lon/2)**2
    c = 2*atan2(sqrt(a), sqrt(1-a))
 
    R = 6371.0
 
    distance = R*c
    if lat1 > lat2: ## manually checks if Values should be negetive 
        distance = -distance
    if lon1 >lon2:
        distance =-distance
    return distance

def Palmyra_obj():
    """Returns cords of Palymra as a point"""
    import shapely as shp
    return shp.points(-162.078333, 5.883611,)

def Palmyra_plot(ax):
    """"Plots Palmyra onto the graph Returns as """
    Palmyra = Palmyra_obj()
    ax.scatter(Palmyra.x,Palmyra.y, marker  = "o", color = "darkgreen", label = "Palmyra", s= 10)
    return ax 

def Kingman_obj():
    """Returns cords of Palymra as a point"""
    import shapely as shp
    return shp.points(-162.41667, 6.3833,)

def Kingmon_plt(ax):
    kingman = Kingman_obj()
    ax.scatter(kingman.x,kingman.y, marker  = "o", color = "darkgreen", label = "Kingman Reef", s =10)
    return ax

def distance_from_Palymra(ds):

    """Adds collum to ds that produces distance of closest point to palmyra
    This is not the closest point the dFad has gone to palmyra only the distance of the nearest gps location. """
    from shapely.ops import nearest_points
    Palmyra = Palmyra_obj()
    ds["Nearest_point"] = nearest_points(ds["geometry"], Palmyra)[0]
    lat1 = np.array(ds["Nearest_point"].y)
    lon1 = np.array(ds["Nearest_point"].x)
    distance = []
    for n in range(0,len(lat1)):
        temp = haversine_dist(lat1[n], lon1[n], Palmyra.y , Palmyra.x)
        distance.append(temp)
    ds["distance_km"] = distance
    return ds

def samplefreq(ds):
    import shapely as shp
    ds["numpoints"] = shp.get_num_points(ds["geometry"])
    ds["SampleFreq"] = ds["Diff_days"]/ds["numpoints"]*24 ## gives average rate of data in hours
    return ds

def add_distance_collumns(data = gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    """Adds distance dfad traveled in x and y directions, also provides a list of all points"""
    delx_list = []
    dely_list = []
    test_data = data
    delx_long = np.array([])
    dely_long = np.array([])

    lines = test_data["geometry"]
    length = lines.shape[0]

    delxs = []
    delys=[]
    x_distkms = []
    y_distkms = []
    xy_distkms = []
    # bigger loop 

    for n in range(0,length):
        singleline = lines[n]
        coords = list(lines[n].coords)

        x_coords = []
        y_coords = []
        for x, y in coords:
             x_coords.append(x)
             y_coords.append(y)

        x_distkm = []
        y_distkm = []
        xy_distkm = []
        

        for n in range(0,len(x_coords)-1): ##this has to be one less because using n+1
            m = int(n+1)
            xdist = haversine_dist_with_negetive(x_coords[n], y_coords[n], x_coords[m],y_coords[n])## gives just the x distance traveled
            ydist = haversine_dist_with_negetive(x_coords[n], y_coords[n], x_coords[n],y_coords[m]) ## gives just the y distance traveled
            xydist = haversine_dist(x_coords[n], y_coords[n], x_coords[m],y_coords[m]) ##gives distance in both xy
            x_distkm.append(xdist) 
            y_distkm.append(ydist)
            xy_distkm.append(xydist)

        delx = np.diff(x_coords)
        dely=np.diff(y_coords)
        x_distkms.append(x_distkm)
        y_distkms.append(y_distkm)
        xy_distkms.append(xy_distkm)

        delxs.append(delx)
        delys.append(dely)

        delx_long = np.append(delx_long, delx)
        dely_long = np.append(dely_long, dely)

    data["x_deg"] = delxs
    data["y_deg"] = delys
    data["x_km"] = x_distkms
    data["y_km"] = y_distkms
    data["xy_km"] = xy_distkms

    heading = np.atan(delx_long/dely_long)
    delx_long = np.cos(heading)
    dely_long = np.sin(heading)

    return data, delx_long, dely_long

def add_time_collumns(data = gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    """Function to add a list of the times of each GPS point with the Form: 2021-07-02-6:00 
    so it can be indexed into .nc file
    This must already use samplefreq()
    Add column name: "timelist"
    """
    samplefrequencies = data["SampleFreq"]
    startdate = data["MinOfDate"]
    minoftimes = data["MinOfTimes"]

    timelists = []
    for i in range(len(data)):
        interval = samplefrequencies[i]
        temptime = startdate[i]
        hrs = (minoftimes[i]%1)*24
        temptime += pd.Timedelta(hours = hrs)
        timelist = []
        timelist.append(temptime)
        numpoints = len(data.at[i,"x_deg"])
        time_delta = pd.Timedelta(hours = interval)
        time_temp = temptime
        for n in range(numpoints):
            time_temp += time_delta
            timelist.append(time_temp)

        timelists.append(timelist)
    data["timelist"] = timelists
    return data
    
def list_of_latlon(data, droplast = True ):

    """Returns a list string of lat and lon points"""
    lines = data["geometry"]

    points = np.empty((1,2))
    for n in range(0,len(lines)):
        dummyline = lines[n]
        cords  = np.array(dummyline.coords)
        if droplast == True:
            cords = cords[:-1]
        points = np.append(points,cords, axis= 0 )

    rotatedpoints = np.rot90(points) 
    #print(rotatedpoints.shape) ##this gives us all of lats in first row and all of the lon in second row 

    lat  = rotatedpoints[0][1:]
    lon = rotatedpoints[1][1:]
    return lat, lon

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

def plotting_zoom(d, ax):
    palmyra = Palmyra_obj()
    palmyra_cords = [palmyra.x, palmyra.y]
    ax.set_xlim(palmyra_cords[0]-d, palmyra_cords[0]+d)
    ax.set_ylim(palmyra_cords[1]-d, palmyra_cords[1]+d)
    ax.tick_params(axis='x', labelrotation=45)   

def nonUnique_tracks(data):
    non_unique = data.loc[data.duplicated(subset= "Name_ID", keep= False)]
    non_unique = non_unique.sort_values(by = "Name_ID")
    return non_unique

def Histogram2dslope(data):
    data = data.set_axis(["Longitude", "Latitude"], axis =1)
    lon = data['Longitude']
    lat = data['Latitude']

    gridded, lonedges, latedges  = np.histogram2d(lon,lat,bins = 100)
    gridded = gridded.T

    lon_avg = np.mean(gridded, axis= 0) ## provided average lon
    lat_avg = np.mean(gridded, axis= 1) ##provides averaged lat 

    max_lon = np.argmax(lon_avg)
    max_lat = np.argmax(lat_avg)
    lat_avg[max_lat] = np.mean(lat_avg)
    lon_avg[max_lon] = np.mean(lon_avg)

    lon_coefficents = np.polyfit(lonedges[:-1], lon_avg, 1)
    lat_coefficents = np.polyfit(latedges[:-1], lat_avg, 1)

    return lon_coefficents, lat_coefficents

def histogram2d(data, bins = 100):
    data = data.set_axis(["Longitude", "Latitude"], axis =1)
    lon = data['Longitude']
    lat = data['Latitude']

    gridded, lonedges, latedges  = np.histogram2d(lon,lat,bins = bins)
    gridded = gridded.T
    return gridded, lonedges, latedges

def Add_x_y_speed_collums(data): 
    """Converts distances in x, y, and xy to speeds in x,y,xy in m/s, using the sampleing frequency"""
    convert = 1000/(60**2) ## km/hr to m/s 

    xspeeds = []
    yspeeds = []
    xyspeeds = []
    for i in data.index:
        xdistance = np.array(data.at[ i, "x_km"])
        xspeed = xdistance/data.at[ i, "SampleFreq"]*convert
        xspeeds.append(xspeed)

        ydistance = np.array(data.at[i, "y_km"])
        yspeed = ydistance/data.at[ i, "SampleFreq"]*convert
        yspeeds.append(yspeed)

        xydistance = np.array(data.at[i,"xy_km"])
        xyspeed = xydistance/data.at[ i, "SampleFreq"]*convert
        xyspeeds.append(xyspeed)

    data["x_speed"] = xspeeds
    data["y_speed"] = yspeeds
    data["xy_speed"] = xyspeeds
    return data

def Add_Mapped_speeds(ds = gpd.GeoDataFrame)->gpd.GeoDataFrame:
    """Produces a column with the Ocean speeds "mapped" onto the cordinates of the dFADs"""
    Mapped_speeds = []
    for i in range(len(ds)):
        ulist = ds["mapped_u"][i]
        vlist = ds["mapped_v"][i]
        speed = np.sqrt(ulist**2+vlist**2)
        Mapped_speeds.append(speed)

    ds["mapped_speed"] = Mapped_speeds
    return ds

def Add_Delta_speeds(ds = gpd.GeoDataFrame):
    """Provides columns deltax_speed, deltay_speed, delta_speed. 
    Which are the diffence between the dFAD speeds and model speeds
    returns the dataset with added columns"""

    delta_xspeeds = []
    delta_yspeeds = []
    delta_speeds = []

    for i in range(len(ds)):
        mapped_x = ds["mapped_u"][i]
        dfad_x = ds["x_speed"][i]
        deltax = mapped_x[:-1] - dfad_x
        #Y values
        mapped_y = ds["mapped_v"][i]
        dfad_y = ds["y_speed"][i]
        deltay = mapped_y[:-1] - dfad_y
        #speed 
        mapped = ds["mapped_speed"][i]
        dfad = ds["xy_speed"][i]
        delta = mapped[:-1] - dfad

        delta_xspeeds.append(deltax)
        delta_yspeeds.append(deltay)
        delta_speeds.append(delta)

    ds["deltax_speed"] = delta_xspeeds
    ds["deltay_speed"] = delta_yspeeds
    ds["delta_speed"] = delta_speeds
    return ds

def Add_interp_currents(data,vo,uo):
    from scipy.interpolate import interpn 
    def closest_point(data, lat, lon, depth, time):
        nearest = data.sel(latitude = lat, longitude = lon, depth = depth,time = time, method = "nearest") 
        return nearest
    
    def second_time(time,nearest ):
        delta = pd.Timedelta(days = 1)
        if time < nearest.time.to_numpy():
            newtime = time -delta
            dayfrac = (time.hour + (time.minute)/60)/24
            dayfrac = 1-dayfrac
        if time > nearest.time.to_numpy():
            newtime = time + delta
            dayfrac = (time.hour + (time.minute)/60)/24
        if time == nearest.time.to_numpy():
            newtime = time
            dayfrac = 1
        return newtime, dayfrac
    
    mapped_vs = []
    mapped_us = []
    y = vo.latitude.to_numpy()
    x = vo.longitude.to_numpy()
    cords = (y,x)

    for i in range(len(data)):
        print(i)
        pointlegnth = len(data.at[i,"TimeStamp"])
        mapped_u = []
        mapped_v = []
        for n in range(pointlegnth):
            time  = data.at[i,"TimeStamp"][n]
            time = pd.to_datetime(time)
            point = sp.get_point(data.at[i,"geometry"],n)
            lat = point.y
            lon = point.x
            poi = (lat,lon)
            ##getting the nearest two points
            ##First V
            valuest1 = vo.sel(depth = 15,time = time, method="nearest").to_numpy()
            p1 = interpn(cords, valuest1,poi)
            nearest = closest_point(vo,lat,lon,15,time)
            newtime, dayfrac = second_time(time,nearest)
            valuest2 = vo.sel(depth = 15,time = newtime, method="nearest").to_numpy()
            p2 = interpn(cords, valuest2,poi)
            values = np.array([p1[0],p2[0]])
            v = np.interp(dayfrac, [0,1], values)
            v = v.astype(float)
            mapped_v.append(v)
            npmapped_v = np.array(mapped_v)
            listmapped_v = npmapped_v.tolist()

            ##Then U
            valuest1 = uo.sel(depth = 15,time = time, method="nearest").to_numpy()
            p1 = interpn(cords, valuest1,poi)
            nearest = closest_point(uo,lat,lon,15,time)
            newtime, dayfrac = second_time(time,nearest)
            valuest2 = uo.sel(depth = 15,time = newtime, method="nearest").to_numpy()
            p2 = interpn(cords, valuest2,poi)
            values = np.array([p1[0],p2[0]])
            u = np.interp(dayfrac, [0,1], values)
            u = u.astype(float)
            mapped_u.append(u)
            npmapped_u = np.array(mapped_u)
            listmapped_u = npmapped_u.tolist()



        mapped_vs.append(listmapped_v)
        mapped_us.append(listmapped_u)
    
    data["mapped_v"] = mapped_vs
    data["mapped_u"] = mapped_us
    return data

def Rolling_mean(x = list,windowsize = int):
    """Insert a list, produces a rolling mean with specified window size
    returns x, the same size"""
    line  = pd.DataFrame({"speed" : x})
    line = line["speed"].rolling(windowsize, center=True,min_periods=1).mean()
    line  = line.to_list()
    return line

def OneTrajectory(ax, ds= gpd.GeoDataFrame,index= int ):
    """For plotting a single Trajectory on specified axis"""
    line = ds.at[index,'geometry']
    if line == None: 
        return ax
    x,y = line.xy
    ax.plot(x,y)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("latitude")
    return ax

def Rolling_avg_no_nan(list, window):
    x = Rolling_mean(list, window)
    x = np.array(x)
    masked = ~np.isnan(x)
    x = x[masked]
    return x

def Add_avg_speed(ds :gpd.GeoDataFrame)->gpd.GeoDataFrame:
    ds.columns
    avg_xspeed = []
    avg_yspeed = []
    avg_speed = []

    for i in range(len(ds)):
        xspeed = ds.at[i,"x_speed"]
        yspeed = ds.at[i,"y_speed"]
        speed = ds.at[i,"xy_speed"]
        #speed1 = (xspeed**2 +yspeed**2)**(1/2)

        xspeed = Rolling_avg_no_nan(xspeed, 6)
        yspeed = Rolling_mean(yspeed,6)
        speed = Rolling_mean(speed,6)
        avg_xspeed.append(xspeed)
        avg_yspeed.append(yspeed)
        avg_speed.append(speed)

    ds["avg_xspeed"] = avg_xspeed
    ds["avg_yspeed"] = avg_yspeed
    ds["avg_speed"] = avg_speed
    return ds

def Add_Delta_speeds(ds = gpd.GeoDataFrame, Average =False, Direction = False):
    """Provides columns deltax_speed, deltay_speed, delta_speed. 
    Which are the diffence between the dFAD speeds and model speeds
    If direction is False Takes absolute of vectors. 
    if average speed is True: Calcuates the rolling speed avage 
    returns the dataset with added columns"""

    delta_xspeeds = []
    delta_yspeeds = []
    delta_speeds = []

    for i in range(len(ds)):
        # X Values
        mapped_x = ds["mapped_u"][i]
        dfad_x = ds["x_speed"][i]
        if Average == True:
            dfad_x = ds["avg_xspeed"][i]

        deltax = mapped_x[:-1] - dfad_x ## Magnitude and direction
        if Direction == False:
            deltax = (np.abs(mapped_x[:-1]) - np.abs(dfad_x)) # comparing magnitude only ## if want to use relative include /(np.abs(dfad_x))
        
        #Y values
        mapped_y = ds["mapped_v"][i]
        dfad_y = ds["y_speed"][i]
        if Average == True:
            dfad_y = ds["avg_yspeed"][i]
        
        deltay = mapped_y[:-1] - dfad_y ## Magnitude and direction
        if Direction == False:
            deltay = (np.abs(mapped_y[:-1]) - np.abs(dfad_y)) # comparing magnitude only
        
        #speed 
        mapped = ds["mapped_speed"][i]
        dfad = ds["xy_speed"][i]
        if Average == True:
            dfad = ds["avg_speed"][i]
        
        delta = mapped[:-1] - dfad ## Magnitude and direction
        if Direction == False: 
            delta = (np.abs(mapped[:-1]) - np.abs(dfad))# comparing magnitude only
    

        delta_xspeeds.append(deltax)
        delta_yspeeds.append(deltay)
        delta_speeds.append(delta)
    if Average == True:
        ds["deltax_speed_avg"] = delta_xspeeds
        ds["deltay_speed_avg"] = delta_yspeeds
        ds["delta_speed_avg"] = delta_speeds
    else:
        ds["deltax_speed"] = delta_xspeeds
        ds["deltay_speed"] = delta_yspeeds
        ds["delta_speed"] = delta_speeds
    return ds

## Gets a list of the delta speeds for the Statistic 
def list_of_delta_speeds(ds, average = False):
    delxs = []
    delys = []
    delspeeds = []
    if average == True:
        for i in range(len(ds)):
            delx = ds["deltax_speed_avg"][i]
            dely = ds["deltay_speed_avg"][i]
            delspeed = ds["delta_speed_avg"][i]
            delxs.extend(delx)
            delys.extend(dely)
            delspeeds.extend(delspeed)
    if average == False:        
        for i in range(len(ds)):
            delx = ds["deltax_speed"][i]
            dely = ds["deltay_speed"][i]
            delspeed = ds["delta_speed"][i]
            delxs.extend(delx)
            delys.extend(dely)
            delspeeds.extend(delspeed)
    return delxs, delys, delspeeds

def bootstrapping_each_box(lat, lon, variable, bins = 10):
    """Returns values and error bars, produces 1000 bootstrapping samples"""
    from scipy.stats import binned_statistic_2d

    Values,xedges, yedges, binnumber = binned_statistic_2d(lat, lon, variable, statistic= "mean", bins = bins, expand_binnumbers = True)
    binnumber = binnumber -1

    ## making empty array to store the speeds into
    list_array = np.empty((bins,bins), dtype= object)
    for i in range(bins):
        for j in range(bins):
            list_array[i,j] = []

    ##Placing speeds into correct list 
    for i in range(binnumber.shape[1]):
        lati = binnumber[0,i]
        loni = binnumber[1,i]
        value = delxs[i]
        list_array[lati,loni].append(value)
    ## Doing the bootstrapping
    nbootstrapps = 1000
    samplelist = []
    for k in range(nbootstrapps):

        meanarray = np.zeros((bins,bins))
        for i in range(bins):
            for j in range(bins):
                speeds = list_array[i,j]
                randomi = np.random.randint(0,len(speeds),len(speeds))
                sampledspeeds = []
                for n in range(len(randomi)):
                    sampledspeeds.append(speeds[randomi[n]])
                meanarray[i,j] = np.mean(sampledspeeds)
        samplelist.append(meanarray)

    samplelist = np.array(samplelist)
    errors = np.percentile(samplelist,95,axis = 0)
    errorbars = errors - Values
   
    return Values, errorbars, xedges, yedges, errors

def Add_bathymetry(fig,ax):
    from matplotlib import cm
    bath = xr.open_dataset(r"Code\Data\bath.nc")
    bath_cmap = cm.get_cmap("Blues_r").copy()
    bath_cmap.set_over('green')
    negative_levels = np.linspace(-10000, 0, 11)
    cbr = ax.contourf( bath["lon"], bath["lat"], bath["elevation"], 
                    linestyle = "-", cmap = bath_cmap, alpha = 0.8, levels = negative_levels, extend = "max")
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

def Column_to_List(data, column:str, idlist = False):
    """Returns a column as a long list of values"""
    ids = []
    long_list = []
    for i in range(len(data)):
        row = data.at[i, column]
        id = [data.at[i, "BuoyName"]]
        id = id*len(data.at[i, "TimeStamp"])
        long_list.extend(row)
        ids.extend(id)
    if idlist == True: 
        return long_list, ids 
    return long_list

def Remove_speeds_high_low(data:gpd.GeoDataFrame):
    """Removes the high speed and low speed points
    Add column of masked array, and number of points removed"""
    bad_points = []
    speeds = []
    Masks = [ ]
    for i in range(len(data)):
        speed = data.at[i, "xy_speed"]
        speed_high = speed>2
        speed_low = speed < 0.001
        masked = speed_high | speed_low
        masked =~ masked
        filtered_speed = speed[masked]
        bad_point = len(speed) - len(filtered_speed)
        
        speeds.append(filtered_speed)
        bad_points.append(bad_point)
        Masks.append(masked)
    
    dataclean = data.copy()
    dataclean["xy_speed"] = speeds
    dataclean["points_removed"] = bad_points
    dataclean["Masked_array"] = Masks
    return dataclean 

def Filter_geometry_obj(row):
    """Removes the same gps points from the bad speed points"""
    coords = np.asarray(row.geometry.coords)
    coords = coords[1:]
    filtered_coords = coords[row['Masked_array']]
    if len(filtered_coords) > 1:
        return sp.geometry.LineString(filtered_coords)
    else:
    # Return an empty geometry if not enough points remain
        return None

def Filter_Rows(row,column):
    """Function implented to apply mask from bad point removal to to other columns
    if column size doesnt line up we remove the later point [1:]"""
    array =np.asarray(row[f"{column}"])
    if len(array) != len(row["Masked_array"]):
        array = array[1:]
    filtered_data = array[row["Masked_array"]]
    if len(filtered_data)>1:
        return filtered_data
    else:
        return None

def remove_outof_domain(data:gpd.GeoDataFrame):
    boundry = sp.Polygon([(-163.75,4.5), (-163.75,7.8), (-160 +2/3, 7.8),(-160 +2/3, 4.5)])
    Values = []
    for n in range(len(data)):
        line = data["geometry"][n]
        lines , mask = split_trajectory_in_domain(line, domain = boundry)
        if len(lines) <1:
            Values.append(n)
    print(Values)
    data = data.drop(Values, axis = 0)
    data = data.reset_index()
    return data

def split_trajectory_in_domain(line: sp.LineString, domain: sp.Polygon):
    """
    Splits a LineString trajectory into sub-trajectories (LineStrings)
    that lie strictly within a polygon domain.

    Rules:
    - Only keeps original vertices (no new boundary points added).
    - Only keeps sub-trajectories with >= 2 inside vertices.
    - Mask marks original vertices that belong to a valid inside sub-trajectory.

    Parameters
    ----------
    line : shapely.LineString
        Input trajectory.
    domain : shapely.Polygon
        Polygon domain.

    Returns
    -------
    sub_trajectories : list of shapely.LineString
        List of sub-trajectories inside the domain.
    mask : np.ndarray (bool)
        Boolean mask for each vertex of the original line.
    """
    coords = np.array(line.coords)
    inside = np.array([domain.contains(sp.Point(p)) for p in coords])

    sub_trajectories = []
    mask = np.zeros_like(inside, dtype=bool)

    current_segment = []
    current_indices = []

    for i, (pt, flag) in enumerate(zip(coords, inside)):
        if flag:
            current_segment.append(pt)
            current_indices.append(i)
        else:
            # if segment ended, check length
            if len(current_segment) >= 2:
                sub_trajectories.append(sp.LineString(current_segment))
                mask[current_indices] = True
            # reset
            current_segment = []
            current_indices = []

    # final check at end
    if len(current_segment) >= 2:
        sub_trajectories.append(sp.LineString(current_segment))
        mask[current_indices] = True

    return sub_trajectories, mask

def remove_no_TimeStamp(data: gpd.GeoDataFrame):
    """BROKEN:  Unclear what this does
    Removes Rows that do not have TimeStamp Data
    """
    data, xdistance, ydistance = add_distance_collumns(data)
    data = data.dropna(subset=["TimeStamp"])
    data = data.reset_index(drop = True)
    return data

def add_delta_time_collums(data:gpd.GeoDataFrame):
    """Adds a collum that has a list of the change in times, Removes Time stamps that are Nan"""
    Times = []
    for i in range(len(data)):
        timelist = data.at[i, "TimeStamp"]
        timelist = pd.to_datetime(timelist, format = r"%Y-%m-%d %H:%M:%S")
        deltatime = np.diff(timelist)/10**9 ## converts from nano seconds
        Times.append(deltatime)
    data["Delta_Timestamps"] = Times
    return data

def Add_x_y_speed_collums_TimeStamp(data): 
    """Converts distances in x, y, and xy to speeds in x,y,xy in m/s, using the sampleing frequency"""
    convert = 1000 ## km/hr to m/s 

    xspeeds = []
    yspeeds = []
    xyspeeds = []
    for i in data.index:
        xdistance = np.array(data.at[ i, "x_km"])
        xspeed = xdistance/data.at[i,"Delta_Timestamps"].astype("float64")*convert
        xspeeds.append(xspeed)

        ydistance = np.array(data.at[i, "y_km"])
        yspeed = ydistance/data.at[i,"Delta_Timestamps"].astype("float64")*convert
        yspeeds.append(yspeed)

        xydistance = np.array(data.at[i,"xy_km"])
        xyspeed = xydistance/data.at[i,"Delta_Timestamps"].astype("float64")*convert
        xyspeeds.append(xyspeed)

    data["x_speed"] = xspeeds
    data["y_speed"] = yspeeds
    data["xy_speed"] = xyspeeds
    return data

def CSV_to_parquet(filelocation): #### REcheck this with Combinsatlinkandmi.ipynb
    """Takes a CSV with at least the column (BuoyName, Latitude, Longitude, Timestamp, MinOfTimes, MaxOfTimes)
    \nReturns: A Geopandas dataframe with trajectories as a linestring, Keeps these same columns and Timestamp is a list of times at each poin
    \nFiltering
    Only sort of filtering this does is remove any trajectory that is just one point"""

    data= pd.read_csv(filelocation, low_memory=False)
    Latitude_list = data.groupby("BuoyName")["Latitude"].apply(list)
    Longitude_list = data.groupby("BuoyName")["Longitude"].apply(list)
    times_list = data.groupby("BuoyName")["Timestamp"].apply(list)
    date_enter = data.groupby("BuoyName")["MinOfTimes"].apply(np.minimum.reduce)
    date_exit = data.groupby("BuoyName")["MaxOfTimes"].apply(np.minimum.reduce)
    BuoyNames = data["BuoyName"].unique()

    print(len(Latitude_list))
    mask = Latitude_list.apply(len) != 1
    Latitude_list = Latitude_list[mask]
    Longitude_list = Longitude_list[mask]
    times_list = times_list[mask]
    date_enter = date_enter[mask]
    date_exit = date_exit[mask]

    BuoyNames_filtered = Latitude_list.index.to_numpy()
    print(len(Latitude_list))

    Latitude_list
    lines = []
    for n in range(len(Latitude_list)):
        line = sp.linestrings(Longitude_list.iloc[n],Latitude_list.iloc[n])
        lines.append(line)


    print(len(BuoyNames), len(lines), len(date_enter), len(date_exit))
    newdata = gpd.GeoDataFrame({"BuoyName":BuoyNames_filtered, "MinOfDate":date_enter, "MaxOfDate": date_exit,
                                 "TimeStamp": times_list, "geometry": lines}, index= None)
    newdata = newdata.reset_index(drop = True)
    return newdata ## This isnt right

def Satlink_live_to_csv(filelocation: str)->pd.DataFrame:
    """Convets the file download from Santlinks dFAD web portal into the standard csv file
    \n Returns: a Pandas dataframe """
    def dms_to_decimal(coord_str: str) -> float:
        """
        Convert coordinate from 'DDºMM.mmmX' format to decimal degrees.
        
        Example: '14º29.352'S' -> -14.4892
        """
        # regex to capture degrees, minutes, and hemisphere
        match = re.match(r"(\d+)º([\d\.]+)'?([NSEW])", coord_str.strip())
        if not match:
            raise ValueError(f"Invalid coordinate format: {coord_str}")
        
        deg = float(match.group(1))
        minutes = float(match.group(2))
        hemi = match.group(3).upper()
        
        decimal = deg + minutes / 60.0
        
        if hemi in ["S", "W"]:
            decimal = -decimal
        
        return decimal
    
    def extract_prefix(filename: str) -> str:
        """
        Extract prefix before '_positions' from a filename.
        
        Example:
        'SLX_404307_positions_20250924_155130.csv' -> 'SLX_404307'
        """
        match = re.match(r"(.+?)_positions", filename)
        if match:
            return match.group(1)
        else:
            raise ValueError(f"Filename not in expected format: {filename}")

    data['Latitude'] = data['Latitude'].apply(dms_to_decimal)
    data['Longitude'] = data['Longitude'].apply(dms_to_decimal)
    data["StoredTime"] = pd.to_datetime(data["StoredTime"])
    data["BuoyName"] = data["source_file"].apply(extract_prefix)
    data = pd.read_parquet(filelocation)
    data['Timestamp'] = data['StoredTime'].dt.tz_localize(None)
    data = data.sort_values("Timestamp")
    data["MinOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("min")
    data["MaxOfTimes"] = data.groupby(["BuoyName"])['Timestamp'].transform("max")
    data = data.sort_values(by = ['MinOfTimes', "Timestamp"])
    data = data.drop_duplicates()
    cols_to_keep=['BuoyName', 'Timestamp', 'Latitude', 'Longitude']
    data = data[cols_to_keep]
    return data 

def dFADS_in_domain(data:gpd.geodataframe)-> pd.DataFrame: 
    """Calculates the amount of dFADS within the region, and produces a Table with a timeseries of number of points per day
     ERROR: dFADs that leave and reenter have the only the final date that they leave and theirfor are 
     considered active when outside of the domain  """
    mindate = data["MinOfDate"].min()
    maxdate = data["MaxOfDate"].max()
    mindate = mindate.floor('D')
    maxdate = maxdate.floor('D')
    entrydates = pd.DataFrame({"Dates":data["MinOfDate"].dt.floor('D')})
    exitdates = pd.DataFrame({"Dates":data["MaxOfDate"].dt.floor('D')})
    entrydates["Value"] = 1
    exitdates["Value"] = -1

    DateRange = pd.date_range(start = mindate,end = maxdate)
    DateRange = pd.DataFrame({"Dates":DateRange})
    combineddates = pd.concat([entrydates,exitdates]).sort_values("Dates")
    daily_changes = combineddates.groupby('Dates')['Value'].soum().reset_index()
    daily_changes = pd.merge(DateRange, daily_changes, on = "Dates", how = "left").fillna(0)
   # print(daily_changes)
    daily_changes["active_dFADs"] = daily_changes['Value'].cumsum()
    #print(daily_changes)
    return daily_changes

def querry_date(data:gpd.GeoDataFrame, date)-> gpd.GeoDataFrame:
    """Returns only dFADs that are active during this time period"""
    data = data.query("MinOfDate <= @date")
    data = data.query("MaxOfDate >= @date")
    return data

class plotting:
    def __init__():
        ...
    