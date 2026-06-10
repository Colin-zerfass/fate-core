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
    nyears = (data.Time.max() - data.Time.min()).total_seconds()/3600/24/365
    longlist = data.copy()

    climat = xr.open_dataset(settings.DATA_DIR / r"drifter_monthlymeans_9c26_64bd_a00e_U1780506785300.nc")
    climat = climat.rename({"latitude" : "lat", "longitude": "lon", "U": "uo", "V": "vo"}) #eU and eV are errors
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
    
    errors = np.percentile(mean_array,95,axis = 0)/nyears
  
    lat_mean2 = np.mean(Binned_data,axis =1)/nyears ##this is non bootstrapping data 
    lat_mean = np.mean(mean_array,axis = 0)/nyears

    error_bar = errors - lat_mean
    error_bar[13] = np.mean(error_bar)
    lat_mean[13] = np.mean(lat_mean)

    longlist['year'] = longlist.Time.dt.year
    longlist['month'] = longlist.Time.dt.month
    month_bins = np.array([1,4,7,10,13])
    longlist['season'] = pd.cut(longlist['month'], month_bins, right = False) # makes it [a,b) months 1-3, 4-6,7-9, 10-12

    def calc_lat_average(longlist, yearly_profiles = True):
        lat_bins = np.arange(4.5,7.76, 0.25)
        longlist['lat_bin'] = pd.cut(longlist.lats, lat_bins, right = False)
        profiles = pd.DataFrame()
        for n, bin in enumerate(longlist.season.unique().dropna().sort_values()):
            season = longlist[longlist['season'] == bin]
            profile =season.groupby('lat_bin', observed=False)['x_speed'].mean()
            profiles[n+1] = profile
            ## also get seasons by year
            if yearly_profiles:
                for yr in np.sort(season.year.dropna().unique()):
                    year_data = season[season['year'] == yr]
                    year_profile = year_data.groupby('lat_bin', observed=False)['x_speed'].mean()
                    profiles[f'{n+1}_{int(yr)}'] = year_profile
        profiles = profiles.reset_index()
        profiles['lats'] = lat_bins[profiles.index] + np.diff(lat_bins)/2
        return profiles

    def standard_error_profiles(profiles):
        for s in range(1, 5):
            year_cols = [c for c in profiles.columns if str(c).startswith(f'{s}_')]
            if not year_cols:
                continue
            yearly_data = profiles[year_cols]
            std = yearly_data.std(axis=1, ddof=1)
            n = yearly_data.notna().sum(axis=1)
            profiles[f'{s}_SE'] = std / np.sqrt(n)
        return profiles
    
    profiles = calc_lat_average(longlist)
    profiles = standard_error_profiles(profiles)

    box7 = climat.sel(lat = slice(0, 10), lon = slice(-163.75, -160.5))
    box7 = box7.mean(dim = "lon")

    zonalm = longlist.x_speed.mean()
    zonalstd = longlist.x_speed.std()
    merdianalm = longlist.y_speed.mean()
    merdianalstd = longlist.y_speed.std()


    fig, axs = plt.subplots(1,2, figsize = (8,5), dpi = 400)
    ax, ax1 = axs
    miny, maxy = [3, 9]
    ax.barh(latedges[1:-1], lat_mean[1:], height=0.075, xerr=error_bar[1:], ecolor="r", capsize=1.5, alpha = 0.75)
    ax.set_ylabel("Latitude")
    ax.set_xlabel(r"number of GPS fixes/ year/ 0.1$^{\circ}$ $^2$")
    ax.set_xlabel(r"GPS fixes year$^{-1}$ (0.1$^{\circ 2}$ bins)")
    ax.set_title("Latitude Variation in Number of dFADs")
    ax.set(ylim = [miny, maxy], xlim = [0,65])
    colors = ['r', 'g', 'b', 'purple']
    titles = ['Jan-Mar' , 'Apr-Jun', 'July-Sep', 'Oct-Dec']
    for i in range(4):
        ax1.plot(profiles[i+1], profiles.lats, c = colors[i], label = titles[i])
        ax1.fill_betweenx(profiles.lats, profiles[i+1] - profiles[f'{i+1}_SE'],
                                  profiles[i+1] + profiles[f'{i+1}_SE'],
                  color=colors[i], alpha=0.1, edgecolor= 'none')
        mean_u = box7.uo[i*3:(i+1)*3,:].mean(dim = "ClimatologicalMonth")
        err_u = box7.eU[i*3:(i+1)*3,:].mean(dim = "ClimatologicalMonth")
        ax1.plot(mean_u, box7.lat, c = colors[i], ls = '--', alpha = 0.5)
        ax1.fill_betweenx(box7.lat, mean_u - err_u, mean_u + err_u, 
                          color= colors[i], alpha=0.1, edgecolor= 'none')

    ax1.set(title = 'Zonal Velocity of dFADs',xlabel = 'Zonal Velocity (m/s)', ylabel= 'Latitude', ylim= [miny, maxy], xlim =[-0.5, 0.7])
    ax1.plot([], [], c='k', ls='--', label='climatology') ## too be added to legend 
    ax1.legend(loc='upper center', bbox_to_anchor=(3.2/7, -0.12),
            fancybox=True, shadow=True, ncol=3)
    # text = f'mean     $\\sigma$    \n zonal: {zonalm:0.2f} $\\pm${zonalstd:0.2f} \n meridonal: {merdianalm:0.2f} $\\pm${merdianalstd:0.2f}'
    # ax1.text(0.95,0.05,text,  transform=ax1.transAxes,
    #             ha="right", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.7, edgecolor="black"))
    for axi in axs: 
        axi.hlines([7.75, 4.5], -10, 100, color = 'k', alpha = 0.4, ls = ':')

    ax1.arrow(0.3, 3.5, -0.2, 0.0,   width=0.05, head_width=0.1, 
        head_length=0.05, length_includes_head=True, color="black")
    ax1.text(0.2, 3.8, " Westward SEC", fontsize=9, ha="center", va="top")


    ax1.arrow(-0.4, 6, 0.2, 0.0,   width=0.05, head_width=0.1, 
        head_length=0.05, length_includes_head=True, color="black")
    ax1.text(-0.3, 6.3, " Eastward NECC", fontsize=9, ha="center", va="top")

    ax.text(40, 7.8, 'Geofenced Boundary', ha = 'center', fontsize = 7, alpha = 0.6)
    fig.tight_layout()
    ax.text(-0.1, 1.1, 'a)', transform=ax.transAxes, fontsize=12, va='top')
    ax1.text(-0.1, 1.1, 'b)', transform=ax1.transAxes, fontsize=12, va='top')
    FIG2name = settings.FIGURES_PAPER_DIR / 'FIG2.pdf'

    fig.savefig(FIG2name, format = 'pdf')
    print(f'saved Figure 1 to : {FIG2name}')

def FIG3():
    print('Starting FIG3')
    from matplotlib.ticker import ScalarFormatter
    import functions.Autocorrelation as auto 

    ds = gpd.read_parquet(settings.dFAD_DATA)
    drifters = gpd.read_parquet(settings.DRIFTER_DATA)
    # Pramaters_____________
    traj_days = [7,7] #7,7]
    us = ['x_speed', 'mapped_u'] #, 'mapped_u', 'mapped_u_30']
    vs = ['y_speed', 'mapped_v'] # , 'mapped_v', 'mapped_v_30']
    method = 'Vector' #Vector or Series 
    datas = []
    ntrajs = []
    bootstrap = True
    n_resamples = 1000
    blocksize = 50
    #________________________
    for i , days in enumerate(traj_days): 
        data, ntraj = auto.calc_autocorrilation(ds, days, Method = method, maxdt = 26, u = us[i], v = vs[i])
        if bootstrap:
            print('starting bootstrapping')
            data = auto.block_bootstrap(data, n_resamples,blocksize)
            #data = auto.interp_results(data)
        datas.append(data)
        ntrajs.append(len(ntraj))
    ## Autocorrilation of drifters 
    data, ntraj = auto.calc_autocorrilation(drifters, 7, Method = method, maxdt= 26, dt = 1)
    if bootstrap: 
        print('starting bootstrapping')
        data = auto.block_bootstrap(data,n_resamples,blocksize)
        data = auto.interp_results(data)
    datas.append(data)
    ntrajs.append(len(ntraj))

    #plotting
    exponetial = False
    colors = ['g', 'b', 'k', 'k']
    labels = ['dFADs', 'GLORYs 15m', 'Drifters'] #'GLORYs_5m', 'GLORYs 30m',
    import matplotlib.gridspec as gridspec
    from scipy.optimize import curve_fit
    def func(x, a,b,c):
        return a*np.exp(-x/b)+c
    
    # fig ,axs = plt.subplots(1,3, figsize = (6,4), dpi = 300)
    # ax0, ax1, ax2 = axs
    fig = plt.figure(figsize=(6,6), dpi = 300)
    gs = gridspec.GridSpec(2, 2, hspace=0.6)
    ax0 = fig.add_subplot(gs[0,:])
    ax1 = fig.add_subplot(gs[1,0])
    ax2 = fig.add_subplot(gs[1,1], sharey = ax1)

    for i,data in enumerate(datas):
        ## also fit exponetials to these 
        xdata = data.index / np.timedelta64(24, 'h')
        popt, _ = curve_fit(func, xdata, data.R, p0=[1, 1, 0], maxfev=5000)
        xfit = np.linspace(0, xdata.max(), 200)
        if exponetial:
            ax0.plot(xfit, func(xfit, *popt), linestyle='--', color=colors[i], alpha=0.8,
                    label=f"{traj_days[i]} Day Fit (T={popt[1]:.2f}d)")
        ax0.plot(data.index/np.timedelta64(24, 'h'), data.R, label = labels[i], zorder = 10, color = colors[i])
        if bootstrap:
            ax0.fill_between(data.index/np.timedelta64(24, 'h'), data['R_025'], data['R_975'], 
                             color = 'r', label = '95th CI', alpha = 0.25, zorder = 1, linewidth = 0 )
    fig.subplots_adjust(hspace=0.35)
    x = np.linspace(0,12, 100)
    ax0.set_xlim(0,7)
    ax0.set_ylabel(r"$R(\tau)$", fontsize = 14)
    ax0.set_xlabel(r"$\tau$ (days)", fontsize = 12)
    ax0.hlines(0,0, 300, color = "k", alpha = 0.25)
    ax0.minorticks_on()
    ax0.set_title(f"Autocorrelation of dFAD and drifter")

    handles, labels = ax0.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax0.legend(by_label.values(), by_label.keys(), fancybox = True, shadow = True)

    method ='fft'
    window ='hann_periodic' #hann_periodic
    psds_dFADs_u = auto.calc_powerspectrum(dFADs=ds, segment_length= 7, method=method, dimention= 'u', window= window)
    psds_drifters_u = auto.calc_powerspectrum(dFADs=drifters, segment_length= 7, interp_dt= 4, method=method, dimention= 'u', window= window)

    psds_dFADs_v = auto.calc_powerspectrum(dFADs=ds, segment_length= 7, method=method, dimention= 'v', window= window)
    psds_drifters_v = auto.calc_powerspectrum(dFADs=drifters, segment_length= 7, interp_dt= 4, method=method, dimention= 'v', window= window)

    mean_dFADs_u = psds_dFADs_u.groupby('freq', observed = False)['PSD'].mean().reset_index()
    mean_drifters_u = psds_drifters_u.groupby('freq', observed = False)['PSD'].mean().reset_index()

    mean_dFADs_v = psds_dFADs_v.groupby('freq', observed = False)['PSD'].mean().reset_index()
    mean_drifters_v = psds_drifters_v.groupby('freq', observed = False)['PSD'].mean().reset_index()

    # Zonal (u) direction
    ax1.plot(mean_dFADs_u.freq[:]*86400, mean_dFADs_u.PSD[:]/86400, label = 'dFADs', color = 'g')
    # ax1.fill_between(std_dFADs_u.freq, mean_dFADs_u.PSD - 10**std_dFADs_u.PSD_LOG,  mean_dFADs_u.PSD  + 10**std_dFADs_u.PSD_LOG, color = 'b', alpha = 0.25)
    ax1.plot(mean_drifters_u.freq[:]*86400, mean_drifters_u.PSD[:]/86400, label = 'Drifters', color = 'k')

    ax1.set_xlim(mean_dFADs_u.freq[0]*86400 ,mean_dFADs_u.freq.iloc[-1]*86400)
    ax1.set_yscale('log')
    ax1.set(xlabel= 'frequency [day$^{-1}$]', ylabel= 'PSD   m$^2$ s$^{-2}$ c.p.d $^{-1}$ ')
    ax1.set_title('Zonal')
    ax1.set_xscale('log')
    ax1.xaxis.set_major_formatter(ScalarFormatter())
    ax1.set_xticks([0.25, 0.5, 1.0 , 2])
    ax1.grid(True, which='both', alpha=0.2)

    # Meridional (v) direction
    ax2.plot(mean_dFADs_v.freq[:]*86400, mean_dFADs_v.PSD[:]/86400, label = 'dFADs', color = 'g')
    ax2.plot(mean_drifters_v.freq[:]*86400, mean_drifters_v.PSD[:]/86400, label = 'Drifters', color = 'k')

    ax2.set_xlim(mean_dFADs_v.freq[0]*86400 ,mean_dFADs_v.freq.iloc[-1]*86400)
    ax2.set_yscale('log')
    ax2.set(xlabel= 'frequency [day$^{-1}$]') #ylabel= 'PSD   m$^2$ s$^{-2}$ Hz$^{-1}$'
    ax2.legend()
    ax2.set_title('Meridional')
    ax2.tick_params(labelleft=False)
    ax2.set_xscale('log')
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.grid(True, which='both', alpha=0.2)
    ax2.set_xticks([0.25, 0.5, 1.0 , 2])

    shared_ax = fig.add_subplot(gs[1, :], frameon=False)
    shared_ax.set_title("Mean Power Spectral Density", pad = 20)
    shared_ax.tick_params(labelcolor='none',top=False, bottom=False, left=False, right=False)
    fig.tight_layout()

    FIG3_name  = settings.FIGURES_PAPER_DIR / 'FIG3.png'
    fig.savefig(FIG3_name)
    print(f'FIG3 saved to: {FIG3_name}')

def Fig_appendex(): 
    drifters = gpd.read_parquet(settings.DRIFTER_DATA)
    longlist = funcs.generate_longlist(drifters)
    maxtime = longlist.Time.max()
    mintime = longlist.Time.min()
    # fig, ax = plt.subplots(1,2)
    #ax0, ax1 = ax.flatten() 
    fig = plt.figure(figsize=(12,4), dpi = 300)
    gs = fig.add_gridspec(1,2, width_ratios = (2.8,1), wspace=0)

    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    
    ax0 = plot.Plotting(drifters, len(drifters), ax0, linewidth=0.75,)
    ax0.get_legend().remove()
    ax0.set_ylim(-8, 35)
    ax0.set_xlim(-210, -100)
    ax0.set_aspect('equal')
    ax0.grid(alpha = 0.25)
    ax0.set_title(f'Drifter Trajectories \n {mintime.date()} - {maxtime.date()}')

    ax1 = plot.Plotting(drifters, len(drifters), ax1 , linewidth=0.75)
    ax1.set_ylim(4.5, 7.75)
    ax1.set_xlim(-163.75, -160.6)
    ax1.set_aspect('equal')
    ax1.grid(alpha = 0.25)
    ax1.get_legend().remove()
    ax1.set_title(f'Drifters within dFAD Geofenced Area')

    FIG_appendex_name = settings.FIGURES_PAPER_DIR/'Appendex1.png'
    fig.savefig(FIG_appendex_name)
    print(f'FIG Appendex 1 saved to : {FIG_appendex_name}')

if __name__ == '__main__':
    # FIG1()
    # FIG2()
    #FIG3()
    Fig_appendex()