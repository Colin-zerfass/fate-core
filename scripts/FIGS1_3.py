"""
Script to reproduce Figures 1 - 3 from the paper 

FIG1 shows all Active dFADs 
- Needs: dFAD data, 
- Optional: Bathymetry Data,

FIG2: Shows Latitude Variations in dFAD along with zonal speed compaired to drifters
- Needs: dFAD Data, Climotogoloty of Drifter 

FIG3: Autocorrilation of dFADs and Drifters 
- Needs: dFAD data, Autocorrilation of drifter data provided by Ria 
"""


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt 
import numpy as np 
import functions.funcs as funcs
import functions.settings as settings
import xarray as xr
import functions.plotting as plot


def FIG1():
    print('Starting FIG1')
    ds = gpd.read_parquet(settings.dFAD_DATA)
    longlist = pd.DataFrame(columns = ['BuoyID', 'TimeStamp', 'lat', 'lon'])

    longlist['TimeStamp'], longlist['BuoyID'] = funcs.Column_to_List(ds, 'TimeStamp', True)
    longlist['lat'], longlist['lon'] = funcs.list_of_latlon(ds, False)
    longlist['TimeStamp'] = pd.to_datetime(longlist['TimeStamp'])

    ## amount of dFAD active on a given day 
    daily = longlist.copy()
    daily['day'] = daily['TimeStamp'].dt.date
    daily_dFAD = daily.drop_duplicates(['BuoyID','day'])
    daily = daily_dFAD.groupby('day', observed= False)['BuoyID'].count()
    daily = daily.reset_index()
    daily = daily.rename(columns={'BuoyID': 'dFADs'})
    daily['year'] = pd.to_datetime(daily.day).dt.year

    target_day = pd.to_datetime("2024-4-25")
    past = longlist[longlist.TimeStamp < pd.to_datetime(target_day)].copy()
    past['day'] = past.TimeStamp.dt.date
    past['last_date'] = past.groupby('BuoyID', observed = False)['day'].transform('last')
    past['last_date'] = pd.to_datetime(past['last_date'])
    activeday = past[past['last_date'] == (target_day - pd.Timedelta(1, 'day'))].copy()
    activeday['deltat'] = activeday.groupby('BuoyID')['TimeStamp'].diff()
    gap = pd.Timedelta(days=2)

    # Build segment IDs that increment at each large time gap
    seg_id = activeday['deltat'].ge(gap).groupby(activeday['BuoyID']).cumsum()

    # Keep only the most recent segment per buoy (or all rows if no gap >= 2 days)
    activeday = activeday[seg_id == seg_id.groupby(activeday['BuoyID']).transform('max')].copy()
    activeday = activeday.set_index(['BuoyID'])

    ## Plotting Data 
    bouyids = activeday.index.unique()

    fig, ax = plt.subplots(figsize = (5,5), dpi = 400)
    fig, ax = plot.Add_bathymetry(fig, ax)

    for id in bouyids: 
        traj = activeday.loc[id]
        if isinstance(traj, pd.Series):
            traj = traj.to_frame().T
        
        traj = traj.sort_values('TimeStamp')

        ax.plot(traj.lon, traj.lat)
        ax.scatter(traj.lon.iloc[-1], traj.lat.iloc[-1])

        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

    ax.set_title(f'dFADs Trajectories on\n {target_day.date()}')

    # Center globe on the mean position of the latest point from each buoy
    last_obs = activeday.sort_values("TimeStamp").groupby(level=0).tail(1)
    center_lon = float(last_obs["lon"].mean())
    center_lat = float(last_obs["lat"].mean())

    # Place a small globe inset in the top-right corner of the existing axis
    bbox = ax.get_position()  # in figure coordinates
    w, h = 0.18, 0.18
    x0 = bbox.x0 -w/2
    y0 = bbox.y1 -h/2

    globe_ax = fig.add_axes([x0, y0, w, h], projection=ccrs.Orthographic(center_lon+10, center_lat))
    globe_ax.set_global()
    globe_ax.add_feature(cfeature.LAND, facecolor="0.9")
    globe_ax.add_feature(cfeature.OCEAN, facecolor="white")
    globe_ax.coastlines(linewidth=0.4)
    globe_ax.gridlines(linewidth=0.2, color="gray", alpha=0.5)

    # Mark the region of interest on the globe
    globe_ax.scatter(center_lon, center_lat, s=15, color="crimson", transform=ccrs.PlateCarree(), zorder=5)
    lon, lat = plot.Palmyra_obj().xy
    pal_lon, pal_lat =lon[0], lat[0]


    ax.annotate( "Palmyra", xy=(pal_lon, pal_lat),xytext=(18, 18),textcoords="offset points",
                arrowprops=dict(arrowstyle="->", lw=1, color="black"),fontsize=10,zorder=8)
    dataNWR = gpd.read_file(settings.DATA_DIR / "Palmyra_Shapefiles",  layer = 'PAL_KING_NWR_12nm')
    ax = plot.plot_NWPs(ax, dataNWR)
    # ax.legend(loc='upper center', bbox_to_anchor=(3.2/7, 0.02),
    #           fancybox=True, shadow=True, ncol=3)
    FIG1filename = settings.FIGURES_PAPER_DIR / "FIG1.pdf"
    fig.savefig(FIG1filename , format = 'pdf')
    print(f'saved Figure 1 to : {FIG1filename}')



def FIG2():
    print('Starting FIG2')
    ds = gpd.read_parquet(settings.dFAD_DATA)

    data  = funcs.generate_longlist(ds)
    longlist = data.copy()

    climat = xr.open_dataset(settings.DATA_DIR / r"drifter_monthlymeans_f43a_401f_20bd_U1775085192326.nc")
    climat = climat.rename({"latitude" : "lat", "longitude": "lon", "U": "uo", "V": "vo"})
    lat_range = np.arange(min(data.lats), max(data.lats), 0.1)
    lon_range = np.arange(min(data.lons), max(data.lons), 0.1)

    Binned_data, lonedges, latedges  = funcs.histogram2d(data, bins = [lon_range,lat_range])

    
    mean_array = []
    np.random.seed(50)
    for n in range(1000):
        Randidex  = np.random.randint(0,30,30)
        selected_bins = []
        for i in Randidex:
            selected_bins.append(Binned_data[:,i])


        selected_bins = np.array(selected_bins)
        mean_array.append(np.mean(selected_bins, axis = 0))


    mean_array = np.array(mean_array)
    
    errors = np.percentile(mean_array,95,axis = 0)/3
  
    lat_mean2 = np.mean(Binned_data,axis =1)/3 ##this is non bootstrapping data 
    lat_mean = np.mean(mean_array,axis = 0)/3

    error_bar = errors - lat_mean
    error_bar[13] = np.mean(error_bar)
    lat_mean[13] = np.mean(lat_mean)

    longlist['year'] = longlist.Time.dt.year
    longlist['month'] = longlist.Time.dt.month
    month_bins = np.array([1,4,7,10,13])
    longlist['season'] = pd.cut(longlist['month'], month_bins, right = False) # makes it [a,b) months 1-3, 4-6,7-9, 10-12

    def calc_lat_average(longlist):
        lat_bins = np.arange(4.5,7.76, 0.25)
        longlist['lat_bin'] = pd.cut(longlist.lats, lat_bins, right = False)
        profiles = pd.DataFrame()
        for n, bin in enumerate(longlist.season.unique().dropna().sort_values()):
            season = longlist[longlist['season'] == bin]
            profile =season.groupby('lat_bin', observed=False)['x_speed'].mean()
            profiles[n+1] = profile
        profiles = profiles.reset_index()
        profiles['lats'] = lat_bins[profiles.index] + np.diff(lat_bins)/2
        return profiles

    profiles = calc_lat_average(longlist)

    box7 = climat.sel(lat = slice(0, 10), lon = slice(-163.75, -160.5))
    box7 = box7.mean(dim = "lon")

    zonalm = longlist.x_speed.mean()
    zonalstd = longlist.x_speed.std()
    merdianalm = longlist.y_speed.mean()
    merdianalstd = longlist.y_speed.std()


    fig, axs = plt.subplots(1,2, figsize = (8,5), dpi = 400)
    ax, ax1 = axs
    miny, maxy = [3, 9]
    ax.barh(latedges[:-1], lat_mean, height=0.075, xerr=error_bar, ecolor="r", capsize=1.5)
    ax.set_ylabel("Latitude")
    ax.set_xlabel(r"number of GPS fixs/ year/ 0.1$^{\circ}$ $^2$")
    ax.set_title("Latitude Variation in number of dFADs")
    ax.set(ylim = [miny, maxy], xlim = [0,90])
    colors = ['r', 'g', 'b', 'purple']
    titles = ['Jan-Mar' , 'Apr-Jun', 'July-Sep', 'Oct-Dec']
    for i in range(4):
        ax1.plot(profiles[i+1], profiles.lats, c = colors[i], label = titles[i])
        ax1.plot(box7.uo[i*3:(i+1)*3,:].mean(dim = "ClimatologicalMonth"), box7.lat, c = colors[i], ls = ':', alpha = 0.5)
    ax1.set(title = 'Zonal Velocity of dFADs',xlabel = 'Zonal Velocity (m/s)', ylabel= 'Latitude', ylim= [miny, maxy], xlim =[-0.5, 0.7])
    ax1.plot([], [], c='k', ls=':', label='climatology') ## too be added to legend 
    ax1.legend(loc='upper center', bbox_to_anchor=(3.2/7, -0.1),
            fancybox=True, shadow=True, ncol=3)
    text = f'mean     $\\sigma$    \n zonal: {zonalm:0.2f} $\\pm${zonalstd:0.2f} \n meridianal: {merdianalm:0.2f} $\\pm${merdianalstd:0.2f}'
    ax1.text(0.95,0.05,text,  transform=ax1.transAxes,
                ha="right", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="black"))
    for axi in axs: 
        axi.hlines([7.75, 4.5], -10, 100, color = 'k', alpha = 0.4, ls = '--')

    ax1.arrow(-0.2, 4.0, -0.2, 0.0,   width=0.05, head_width=0.1, 
        head_length=0.05, length_includes_head=True, color="black")
    ax1.text(-0.3, 4.3, "SEC", fontsize=9, ha="center", va="top")


    ax1.arrow(-0.4, 6, 0.2, 0.0,   width=0.05, head_width=0.1, 
        head_length=0.05, length_includes_head=True, color="black")
    ax1.text(-0.3, 6.3, "NECC", fontsize=9, ha="center", va="top")

    ax.text(40, 7.8, 'Geofenced Boundry', ha = 'center', fontsize = 7, alpha = 0.6)
    fig.tight_layout()
    ax.text(-0.1, 1.1, 'a)', transform=ax.transAxes, fontsize=12, fontweight='bold', va='top')
    ax1.text(-0.1, 1.1, 'b)', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top')
    FIG2name = settings.FIGURES_PAPER_DIR / 'FIG2.pdf'

    fig.savefig(FIG2name, format = 'pdf')
    print(f'saved Figure 1 to : {FIG2name}')

def FIG3():
    print('Starting FIG3')
    import functions.Autocorrelation as auto 

    ds = gpd.read_parquet(settings.dFAD_DATA)
    traj_days = [7]
    method = 'Vector' #Vector or Series 
    datas = []
    ntrajs = []
    bootstrap = True
    n_resamples = 1000
    for days in traj_days: 
        data, ntraj = auto.calc_autocorrilation(ds, days, Method = method, maxdt = 26)
        if bootstrap == True:
            print('starting bootstrapping')
            samples = []
            for s in range(n_resamples): 
                sample_index = np.random.choice(data.index.max()+1, data.index.max()+1, replace = True)
                sample = data.loc[sample_index]
                sample = sample.groupby('Tau').mean()
                sample = sample.reset_index(drop = False)
                samples.append(sample)
            data = pd.concat(samples)
        groupeddata  = data.groupby("Tau").mean()
        groupeddata['95th'] = data.groupby('Tau').quantile(0.95)
        groupeddata['5th'] = data.groupby('Tau').quantile(0.05)
        datas.append(groupeddata)
        ntrajs.append(len(ntraj))


    from scipy.optimize import curve_fit
    def func(x, a,b,c):
        return a*np.exp(-x/b)+c

    exponetial = True
    plot_drfiters = True

        
    import matplotlib.gridspec as gridspec
    # fig = plt.figure(figsize=(6,3), dpi = 300)
    # gs = gridspec.GridSpec(1, 2, width_ratios=[3,1])
    # ax0 = fig.add_subplot(gs[0,0])
    fig ,ax0 = plt.subplots(figsize = (6,4), dpi = 300)

    labels_uivi = ['zonal', 'meridianal']
    colors = ['g', 'k']
    if method == 'Vector': 
        # for i,data in enumerate(datas_ui):
        #     ax0.plot(data.index/np.timedelta64(24, 'h'), data.R, label =labels_uivi[i])
        for i,data in enumerate(datas):
            ## also fit exponetials to these 
            xdata = data.index / np.timedelta64(24, 'h')
            popt, _ = curve_fit(func, xdata, data.R, p0=[1, 1, 0], maxfev=5000)
            xfit = np.linspace(0, xdata.max(), 200)
            if exponetial:
                ax0.plot(xfit, func(xfit, *popt), linestyle='--', color=colors[i], alpha=0.8,
                        label=f"{traj_days[i]} Day Fit (T={popt[1]:.2f}d)")
            ax0.plot(data.index/np.timedelta64(24, 'h'), data.R, label = f"{traj_days[i]} Day Segment", zorder = 10, color = colors[i])
            if bootstrap:
                ax0.fill_between(data.index/np.timedelta64(24, 'h'), data['95th'], data['5th'], color = 'r', label = 'Errors 5th-95th', alpha = 0.5, zorder = 1)
    if method == 'Series':
        for i, data in enumerate(datas):
            # ax0.plot(data.index/np.timedelta64(24, 'h'), data.Ru, label = "zonal")
            # ax0.plot(data.index/np.timedelta64(24, 'h'), data.Rv, label = "meridianal")
            ax0.plot(data.index/np.timedelta64(24, 'h'), data.Rspeed, label = f"{traj_days[i]} Day Segment")
    if plot_drfiters: 
        dr = pd.read_csv(settings.DATA_DIR / 'drifter_ri_downsampled_7days.csv')
        # dr['hours'] = pd.(dr['hours'], unit= )
        ax0.plot(dr.hours/24, dr.autocorr_ri, label = f"{traj_days[i]} Day drifter Segment", zorder = 10, color = 'k')
        ax0.fill_between(dr.hours/24, dr.ci_low, dr.ci_high, color = 'r', alpha = 0.5, zorder = 1)
        xdata = dr.hours/24
        popt_drifter, _ = curve_fit(func, xdata, dr.autocorr_ri, p0=[1, 1, 0], maxfev=5000)

    x = np.linspace(0,12, 100)
    ax0.set_xlim(0,7)
    ax0.set_ylabel(r"$R(\tau)$", fontsize = 14)
    ax0.set_xlabel(r"$\tau$ (days)", fontsize = 12)
    ax0.hlines(0,0, 300, color = "k", alpha = 0.25)
    ax0.minorticks_on()
    ax0.set_title(f"Autocorrelation of dFAD and drifter")


    # ax1 = fig.add_subplot(gs[0,1])
    # ax1.axis("off")
    # ax1.text(-0.3,0.5, r"$R_i(\tau) = \frac{\overline{\left(U_i(t) - \overline{U}_i\right)\cdot\left(U_i(t+\tau) - \overline{U}_i\right)}}{\sigma_i^2}$",
    #           fontdict={"fontsize" : 15})
    # ax1.text(-0.3,0.9, f"dFAD Trajectory segements:\n" + "".join(f"{a}" for a in ntrajs),
    #           fontdict={"fontsize" : 15})
    # ax1.text(-0.3,0.75, f"Segment length:\n" + "".join(f"{a}" for a in traj_days), #traj_days 
    #           fontdict={"fontsize" : 15})
    # ax0.text(0.6,0.6, fr"drifter $\tau$: {popt_drifter[1]:.3f} days" + '\n' rf"dFADs $\tau$: {popt[1]:.3f} days",
    #           fontdict={"fontsize" : 13}, transform = ax0.transAxes)
    fig.tight_layout()
    ax0.legend(fancybox = True, shadow = True)
    FIG3_name  = settings.FIGURES_PAPER_DIR / 'FIG3.pdf'
    fig.savefig(FIG3_name)
    print(f'FIG3 saved to: {FIG3_name}')

if __name__ == '__main__':
    # FIG1()
    # FIG2()
    FIG3()