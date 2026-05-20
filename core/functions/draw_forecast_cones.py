import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from functions.funcs import *
import functions.output_functions as output 

from shapely.geometry import LineString, Point
from matplotlib.path import Path
"""
plotting tools to add forecasts cones to a forecast trajectory. 

Main function is plot_Forcasts()



"""




def draw_forecast_cone(treudata, Forecast, q, ax): 
    """True data is all the true data within the window that is going to be plotted 
    Forecast is the one particle forecast 
    q is dataframe of uncurtainities to be plotted with collumns"""
    ## methos is to use a shapely object and then plot that. 
    ## start by making a circle at the last point
    Forecast = Forecast.sort_values("leadtime").reset_index(drop = True)
    q70 = q.sel(q = 0.7, method = 'nearest')
    qdf = pd.DataFrame({'qsolved':q70.qsolved,'leadtime' : q70.leadtime})
    ## now do this for all previous points and take the outside
    circles = []
    for n in range(len(Forecast)):
        poi = sp.Point( Forecast.iloc[n]['lon_forcast'], Forecast.iloc[n]['lat_forcast'])
        time = Forecast.iloc[n]['leadtime']
        maxid = np.abs((qdf.leadtime - time)).idxmin()
        r = qdf.qsolved[maxid]
        circle = poi.buffer(r/111)
        circles.append(circle)

    poly = sp.union_all(circles)
    if poly.geom_type == "MultiPolygon":
        for p in poly.geoms:
            x,y = p.exterior.xy
            ax.plot(x,y, color = "r")
            ax.fill(x,y, color = 'r', alpha= 0.1, zorder = 100)
            ax.set_clip_path
    else:
        x,y = poly.exterior.xy
        ax.plot(x,y, color = "r")
        ax.fill(x,y, color = 'r', alpha= 0.1, zorder = 100)
    return ax

def draw_forecast_cone_dynamic(truedata, Forecast, q, ax, res=200, cmap=plt.cm.YlOrRd, contours=True):
    Forecast = Forecast.sort_values("leadtime").reset_index(drop=True)
    
    Forecastlast = Forecast.tail(1).reset_index(drop = True) 

    # build shapely circles and union (your existing logic)
    circles = []
    for n in range(len(Forecast)):
        poi = Point(Forecast.iloc[n]['lon_forcast'], Forecast.iloc[n]['lat_forcast'])
        time = Forecast.iloc[n]['leadtime']
        r = q.sel(leadtime = time, q = 0.9, method = 'nearest').qsolved.values ## could draw other circles too or maybe still draw cone and just do shading 
        circles.append(poi.buffer(r / 111.0))
    poly = sp.union_all(circles)

    # bounds for the raster
    minx, miny, maxx, maxy = poly.bounds
    xs = np.linspace(minx, maxx, res)
    ys = np.linspace(miny, maxy, res)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.column_stack([xv.ravel(), yv.ravel()])

    # use only the last forecast point
    px, py = Forecastlast.iloc[0][['lon_forcast', 'lat_forcast']]
    # raster points already built as `pts` (P,2)
    dists = np.hypot(pts[:, 0] - px, pts[:, 1] - py).reshape(res, res)
    # radius for last point (km -> degrees) and guard against zero
    r90 = float(q.sel(leadtime=Forecastlast.leadtime[0], q=0.9, method='nearest').qsolved.values)
    r_deg = r90 / 111.0
    if r_deg <= 0:
        alpha = np.zeros((res, res))
    else:
        alpha = np.clip(1.0 - dists / r_deg, 0.0, 1.0)

    # tweak contrast if desired
    alpha = alpha**0.8
    # build RGBA image from colormap and use alpha channel
    rgba = cmap(alpha)
    rgba[..., -1] = alpha
    # mask outside union polygon(s) (reuse your Path mask code)
    mask = np.zeros(alpha.shape, dtype=bool)
    polys = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]
    for p in polys:
        path = Path(np.column_stack(p.exterior.xy))
        mask |= path.contains_points(pts).reshape(res, res)
    rgba[~mask] = (0, 0, 0, 0)
    ax.imshow(rgba, extent=(minx, maxx, miny, maxy), origin='lower', interpolation='bilinear', zorder=3)

    qlist = [0.25,0.5, 0.7, 0.9]
    lss = [ '--', '--', '--', '--']
    for i, a in enumerate(qlist):
        circles = []
        for n in range(len(Forecast)):
            poi = Point(Forecast.iloc[n]['lon_forcast'], Forecast.iloc[n]['lat_forcast'])
            time = Forecast.iloc[n]['leadtime']
            r = q.sel(leadtime = time, q = a, method = 'nearest').qsolved.values
            r = max(r, 0.01)
            circles.append(poi.buffer(r / 111.0))
        poly = sp.union_all(circles)
        polys = poly.geoms if poly.geom_type == "MultiPolygon" else [poly]

        # draw all sub-polygon outlines
        for p in polys:
            x, y = p.exterior.xy
            ax.plot(x, y, color="k", lw=0.8, ls=lss[i])

        # label at the max-latitude point of the last circle
        if contours:
            cx_arr = np.array(circles[-1].exterior.xy[0])
            cy_arr = np.array(circles[-1].exterior.xy[1])
            idx = np.argmax(cy_arr)
            ax.text(cx_arr[idx], cy_arr[idx], f"{int(a*100)}%", fontsize=6, ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0, edgecolor='none', pad=1))

    return ax


def interp_ds(ds,dt_hours = 0.5, forecastds = True):
    newlt = np.arange(min(ds.leadtime), max(ds.leadtime)+0.001, dt_hours)
    newlt = np.append(newlt, ds.leadtime)
    newlt = np.sort(newlt)
    newlt = np.unique(newlt)
    ds = ds.set_index('leadtime')
    ds = ds[~ds.index.duplicated()]
    ds = ds.reindex(newlt)
    if forecastds == True:
        ds['lon_forcast'] = ds['lon_forcast'].interpolate(method = 'linear')
        ds['lat_forcast'] = ds['lat_forcast'].interpolate(method = 'linear')

    else:
        ds['lon_true'] = ds['lon_true'].interpolate(method = 'linear')
        ds['lat_true'] = ds['lat_true'].interpolate(method = 'linear')

    ds = ds.reset_index(names = 'leadtime')
    return ds

def plot_Forcasts(BuoyID:str, dFAD_data, dsforcast: list,startday: int, labels: list, 
                  fig, ax, q:np.array, forcastlength = pd.Timedelta(days= 2), 
                  pastTrajectory = pd.Timedelta(days = 3), shaded_cone = False, used_forecast = False): ## could add getting startime and just get nearest point from that.
    
        # Work on copies so the caller's lists are never mutated
    dsforcast     = list(dsforcast)
    labels        = list(labels)
    used_forecast = list(used_forecast) if used_forecast is not False else False
    ## getting true data
    perplot = dsforcast[0].query("BuoyID == @BuoyID ") 
    ##SLX+487116 #10 had loops ##16 weird points ##119 ##hit palyra SLX+463917
    truedata = True_dFAD_data(dFAD_data, BuoyID)
    forcasttimes = perplot.query("leadtime == 0 ").reset_index(drop= True)
    starttime_not_rounded = forcasttimes.at[startday, "Time"]
    starttime = pd.to_datetime(starttime_not_rounded)
    starttime = starttime.round('min')
    truedata['leadtime'] = truedata.DateTime - pd.to_datetime(starttime_not_rounded)
    truedata['leadtime'] = truedata.leadtime.dt.total_seconds()/3600
    truedata = interp_ds(truedata, forecastds= False)
    truedata['DateTime'] = starttime_not_rounded + pd.to_timedelta(truedata.leadtime, unit='hours')
    truedata = truedata[truedata.DateTime < (starttime + forcastlength)]
    truedata = truedata[truedata.DateTime > (starttime - pastTrajectory)]
    ##set x and y lims 
    #fig, ax = plt.subplots(figsize = (6,6))
    forcastlengthhr = forcastlength.total_seconds()/3600
    ax.plot(truedata.lon_true, truedata.lat_true, label = "dFAD Trajectory", lw= 1.5, color = "k", zorder = 6)
    #colors = ["limegreen", "orange", "firebrick", "orange","limegreen"]
    ##Get forcast from that starttime 
    if used_forecast is not False: 
        ##Should be a list of model names ex: ['merged', 'cmems', 'OSCAR']
        print("removing 1 model")
        ds = dsforcast[used_forecast.index('merged')]
        ds = ds.query(f"BuoyID == @BuoyID")
        ds = ds.query(f"starttime == @starttime").reset_index(drop = True)
        model_used = ds.at[0,'best_model']
        del dsforcast[used_forecast.index(model_used)]
        del labels[used_forecast.index(model_used)]
        del used_forecast[used_forecast.index(model_used)]

    colors = ['limegreen', 'magenta', 'cyan']
    for i,ds in enumerate(dsforcast):
        ds = ds.query(f"BuoyID == @BuoyID")
        ds = output.add_starttime(ds)
        ds = ds.query(f"starttime == @starttime").reset_index(drop = True)
        if len(ds) == 0: 
            continue
        ds = interp_ds(ds)
        ds = ds.query('leadtime <= @forcastlengthhr').reset_index(drop = True)
        if len(ds)== 0:
            print('len is 0')
        starty =  ds.at[0,"lat_true"]
        startx =  ds.at[0,"lon_true"]
        ax.plot(ds.lon_forcast, ds.lat_forcast, label= labels[i], alpha = 0.75, color = colors[i] ,zorder =6, lw = 2.5)
        if shaded_cone == False and i == 0 :
            ax = draw_forecast_cone(truedata, ds, q, ax)
        if shaded_cone == True and i == 0 : 
            ax = draw_forecast_cone_dynamic(truedata,ds,q,ax, contours= True)
    deg = 0.5
    #ax.set_ylim([starty -deg, starty +deg+0.2])
    #ax.set_xlim([startx -deg, startx+deg])
    ax.plot(startx, starty, color = "k", lw = 10, alpha= 1,zorder =7)
    ax.set_title(f"{forcastlength.days} day Forecasts of dFAD \n{starttime}")
    return fig, ax   



def plot_circle_km(ax, radius_km=5, **patch_kwargs):
    from matplotlib.patches import Circle
    point = Palmyra_obj()
    circle = Circle(
        (point.x, point.y),
        radius_km,
        fill=False, 
        alpha = 0.75,
        **patch_kwargs
    )
    ax.add_patch(circle)
    return ax



def plot_Forecast_from_dFAD_index(Forecast_data:list, dFAD_data, qdata , dFAD_Index:int, startday_int: int, fig, ax):
    from functions.plotting import Add_bathymetry
    merged = Forecast_data[0]
    merged = output.add_starttime(merged)
    merged = merged.sort_values(['BuoyID', 'starttime', 'Time']).reset_index(drop = True)
    IDs = merged.BuoyID.unique()
    sd = startday_int
    buoyID = IDs[dFAD_Index]

    #getting intial angle and speed to solve error
    Forecast = merged.query(f"BuoyID == @buoyID")
    startdays = Forecast.query(f'leadtime == 0').reset_index(drop = True)
    startday = startdays.at[sd,'Time'].round('min')
    Forecast = Forecast.query(f'starttime == @startday').reset_index(drop= True)
    leadtimes = np.arange(Forecast.leadtime.min(), Forecast.leadtime.max(), 0.5)
    isd = Forecast.initial_speed_dif_mag[0]
    ilat = Forecast.initial_lat[0]
    print(ilat, isd)
    leadtimes = leadtimes[1:]
    ## solving Q
    qs = qdata.copy()
    qs = qs.interp(leadtime = leadtimes, kwargs={"fill_value": "extrapolate"})
    qs['qsolved'] = qs.initial_lat*ilat + qs.initial_speed_dif_mag*isd + qs.Intercept

    fig, ax = plot_Forcasts(buoyID, dFAD_data, Forecast_data, startday = sd, 
                            labels = ["All Methods",'cmems', 'OSCAR'], fig= fig,
                            ax = ax, q=qs, forcastlength= pd.Timedelta(days = 3), 
                            pastTrajectory = pd.Timedelta(days = 7), shaded_cone=True,
                            used_forecast= False) #['merged', 'cmems', 'OSCAR'] 

    fig, ax = Add_bathymetry(fig, ax, colorbar = False)
    ax.set_aspect("equal")
    ax = plot_circle_km(ax, radius_km= 0.0833)
    ax.set(xlim = [-163.75, -160.66], ylim = [ 4.5,7.75])
    fig.tight_layout()
    ax.text(0.02, 0.02, f"initial values\n speed differance: {isd:.2f}\n latitude: {ilat:.2f}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="black"))
    return fig, ax