

import numpy as np 
import pandas as pd 
import geopandas as gpd
import functions.funcs as funcs 
import functions.output_functions as opf
import xarray as xr
import os
import functions.settings as settings

"""Genrates Regression for forecast errors. Error(qauntile, leadtime, initial_speed_dif, latitude)
"""



## loading the data
ds = gpd.read_parquet(settings.dFAD_DATA) 
Forecast_data = settings.FORECAST_DIR / 'cmems_bias_pers_meanremoved_2026.csv'
fc = pd.read_csv(Forecast_data)
output_data = settings.DATA_DIR / 'regression_quantiles_leadtimes_cmems_bias_pers_2026.nc'
plot_outputpath = settings.FIGURES_PAPER_DIR 


fc["Time"] = pd.to_datetime(fc["Time"])
fc['error_km'] = funcs.haversine_df(fc, "lat_true", "lon_true", "lat_forcast", "lon_forcast")
## Unpacking True dFAD data into one list
longlist = funcs.generate_longlist(ds, extra_columns=['mapped_v', 'mapped_v_oscar', 'mapped_u', 'mapped_u_oscar'])
longlist = longlist.rename(columns={'mapped_v': 'v_mapped', 'mapped_v_oscar': 'v_mapped_OSCAR',
                                    'mapped_u': 'u_mapped', 'mapped_u_oscar': 'u_mapped_OSCAR'})

merged = opf.merge_forecast_true(fc, longlist)
merged = opf.add_starttime(merged)
merged["speed"] = np.sqrt(merged.x_speed**2 + merged.y_speed**2)
merged = opf.calc_intial_speed_dif(merged)
merged = opf.calc_iniial_lat(merged)

import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter


if True: # FIG 4

    data= {
        'No Forecast':     ['No_forecast', 'forestgreen'],
        # 'Climatology':      ['climatological2024', 'dodgerblue'],
        'Persistence':      ['persistence', 'blueviolet'],
        # 'All Methods':      ['Final/' + 'Initial_angle_v1_2026', 'k'],
        'GLORYs':             ['CMEMS_2026', 'orange'],
        'GLORYs+bias' :       ['Final/cmems_bias_meanremoved_2026', 'red'],
        'GLORYs+bias+Persistence' : ['cmems_bias_pers_meanremoved_2026', 'black'],
        # 'GLORYs+bias+wind' :       ['cmems_bias_wind_meanremoved_2026', 'green'],
        # 'GLORYs+wind_meanstillon':        ['Final/' + 'cmems_wind_2026', 'olive'], 
        #'GLORYs+wind+pers':   ['Final/'+'CMEMS_wind_pers_2026', 'green'], 
        # 'OSCAR':            ['Final/'+'OSCAR_2026', 'blue'],
        # 'OSCAR+Wind':       ['OSCAR_2022_2025_wind', 'mediumpurple'],
        #'OSCAR+Wind+Pers':  ['Final/'+'OSCAR_wind_pers_2026', 'indigo'],
        #'Initial_speed_dif':    ['Initial_angle_No_pers_2026', 'k']
    }

    # Access all colors
    colors = [v[1] for v in data.values()]

    dslist  = []
    for key, item in data.items(): 
        ds = pd.read_csv( settings.FORECAST_DIR / (item[0]+'.csv'))
        dslist.append(ds)

    ## fix persistance 
    # if 'Persistence' in data.keys():
    #     pers_idx = list(data.keys()).index('Persistence')
    #     persistance = dslist[pers_idx]
    #     persistance = persistance.drop(columns= ["BouyID", 'speed_ms_persistence'])

    #     # to fix persistance column names intial
    #     persistance=persistance.rename(columns={"Latitude_true" : "lat_true", "Longitude_true": "lon_true", 
    #                         "Latitude_persistence": "lat_forcast",
    #                         "Longitude_persistence": "lon_forcast", 
    #                         "lead_time_hours": "leadtime" , "DateTime": "Time"})
    #     dslist[pers_idx] = persistance

    bins = np.linspace(0,8*24,2*24+1)
    skill_score = True ## False: does not interpolate 
    recalc_skill_score = False #(True: takes a 5-10 minutes to run)
    #Outputs to be plotted
    ltes = []
    stds = []
    stds_low = []
    sss = []
    dslist_notinterp = []

    for i, dsi in enumerate(dslist):
        dsi = opf.add_starttime(dsi)
        dsi['error_km'] = opf.haversine_df(dsi, "lat_true", "lon_true", "lat_forcast", "lon_forcast")
        dsi["lead_bins"] = pd.cut(dsi["leadtime"], bins, right= True)
        dslist_notinterp.append(dsi) 

        ltei = dsi.groupby("lead_bins", observed=False).apply(opf.calculate_rmse, include_groups=False).to_numpy()
        pad = np.array([0])
        ltei  = np.concat([pad, ltei])
        ltes.append(ltei)
        
        stdi = dsi.groupby("lead_bins", observed=False)["error_km"].quantile(0.9)
        stdi = np.concat([pad, stdi])
        stds.append(stdi)
        stdi_low = dsi.groupby("lead_bins", observed=False)["error_km"].quantile(0.3)
        stdi_low = np.concat([pad, stdi_low])
        stds_low.append(stdi_low)

        if recalc_skill_score: 
            ##Interpolates Data onto regular intervals
            print(f'calcuating skill score {i}')
            dsi = opf.add_starttime(dsi)
            dsi = dsi.groupby(['BuoyID','starttime'], observed= False).apply(opf.interpolate_data, include_groups = False).reset_index(level=['BuoyID', 'starttime']).reset_index(drop=True)
            dsi = dsi.groupby(['BuoyID','starttime'], observed= False).apply(opf.dtrue, include_groups = False).reset_index(level=['BuoyID', 'starttime']).reset_index(drop=True)
            dsi['dlat_true_km'] = dsi['dlat_true']*110
            dsi['dlon_true_km'] =dsi['dlon_true']*110
            dsi["lead_bins"] = pd.cut(dsi["leadtime"], bins, right= True)
            # Calc Error_km and RMSE of displacement 
            dsi['error_km'] = opf.haversine_df(dsi, "lat_true", "lon_true", "lat_forcast", "lon_forcast")
            dsi["lead_bins"] = pd.cut(dsi["leadtime"], bins, right= True)

            g = dsi.groupby(['BuoyID', 'starttime'], observed=False)
            # cumsumdistance: d is row-wise, dcum is cumsum of d per group
            dsi['d']    = (dsi['dlat_true_km']**2 + dsi['dlon_true_km']**2)**(1/2)
            dsi['dcum'] = g['d'].transform('cumsum')
            # cumsumerror: cumsum of error_km per group
            dsi['error_cum'] = g['error_km'].transform('cumsum')
            # skillscore
            dsi['skillscore'] = 1- (dsi['error_km'] / dsi['dcum'])
            dsi['skillscore'] = dsi['skillscore'].clip(lower = 0)

            ss = dsi.groupby("lead_bins", observed=False)['skillscore'].mean().to_numpy()
            pad = np.array([0])
            ss  = np.concat([pad, ss])
            sss.append(ss)
            dslist[i] = dsi
            # saving so doesnt have to recalulate
    if recalc_skill_score: 
        sss_df = pd.DataFrame(np.stack(sss).T, index=bins, columns=list(data.keys()))

        sss_df.index.name = 'leadtime_hrs'


        sss_df.to_csv(settings.DATA_DIR / 'skill_scores.csv')   
        print(f"Saved skill scores to {settings.DATA_DIR / 'skill_scores.csv'}")

    ## Load saved skill scores (avoids re-running the 5-10 min interpolation)
    sss_df = pd.read_csv(settings.DATA_DIR / 'skill_scores.csv', index_col='leadtime_hrs')
    sss = [sss_df[col].to_numpy() for col in sss_df.columns]

    errorbars = True
    fig = plt.figure(figsize=(10,10), dpi = 400)
    gs = gridspec.GridSpec(3, 2, width_ratios=[3,3], height_ratios= [5,5,3])
    labels = list(data.keys())
    bins = np.linspace(0,8*24,2*24+1)
    if skill_score == True: 
        ax0 = fig.add_subplot(gs[:2,0])
        ax1 = fig.add_subplot(gs[:2,1])
        ax2 = fig.add_subplot(gs[2,:])
        #ax3 = fig.add_subplot(gs[2,1])
    else:     
        ax0 = fig.add_subplot(gs[:,0])
        ax1 = fig.add_subplot(gs[:,1])

    for i in range(len(dslist)):
        # asymmetric error bars: 30th quantile below, 70th quantile above RMSE
        yerr0 = [np.maximum(ltes[i][:13] - stds_low[i][:13], 0), np.maximum(stds[i][:13] - ltes[i][:13], 0)]
        yerr1 = [np.maximum(ltes[i][:] - stds_low[i][:], 0), np.maximum(stds[i][:] - ltes[i][:], 0)]
        alpha = 1
        ls = '-'
        if labels[i] in ['No Forecast', 'Climatology']:
            yerr0 = None
            yerr1 =  None
            alpha = 0.5
            ls = '--'
        if errorbars: 
            ax0.errorbar(bins[0:13], ltes[i][:13], lw = 2, yerr = yerr0, color = colors[i],
                        capsize = 6  , errorevery=4, elinewidth= 0.5, mew = 3, alpha = alpha, ls = ls)
            ax1.errorbar(bins[:]/24, ltes[i][:], lw =2, yerr = yerr1 , label = labels[i],  color =  colors[i],
                        capsize = 6  , errorevery=9, elinewidth= 0.5, mew = 3, alpha =  alpha, ls = ls)
        if not errorbars: 
            ax0.plot(bins[0:13], ltes[i][:13], lw = 2,  color = colors[i], alpha = alpha, ls = ls)
            ax1.plot(bins[:]/24, ltes[i][:], lw = 2,  color = colors[i], alpha = alpha, ls = ls)

        if skill_score == True: 
            ax2.plot(sss_df.index / 24, sss_df.iloc[:, i], color=colors[i], lw=2)

    #limits
    ax0.set_xticks(np.linspace(0,52,14))
    ax0.set_xlim(0,50)
    ax0.set_ylim(0,75)
    #ax0.set_ylim(0,150)
    ax1.set_xlim(0,7.1)
    ax1.set_ylim(0,240)
    ax1.set_xticks(np.linspace(0,7,8))

    ax0.set_ylabel("Dispacement error (RMSE) (km)")
    ax0.set_xlabel("Lead Time (hrs)")
    ax1.set_xlabel("Lead Time (days)")
    ax0.grid(alpha = 0.1)
    ax1.grid(alpha = 0.1)

    # Draw on ax1 the same data-window shown in ax0 (ax0 x is hours, ax1 x is days)
    x0_hr, x1_hr = ax0.get_xlim()
    y0, y1 = ax0.get_ylim()

    zoom_box = plt.Rectangle((x0_hr / 24.0, y0), (x1_hr - x0_hr) / 24.0, y1 - y0,
                            fill=False, edgecolor="black", linestyle="--", linewidth=1.5, zorder=10)
    ax1.add_patch(zoom_box)
    if skill_score == True:
        ax2.set(xlim = [0,7.15],ylim = [-0.1,1], ylabel = 'Skill Score', xlabel ='days')
        ax2.grid(alpha = 0.1)

    fig.legend(loc='upper center', bbox_to_anchor=(3.2/7, 0.08),
            fancybox=True, shadow=True, ncol=3)
    fig.suptitle("Forcasting Position Errors by method \n dFADs 2022/01 - 2026/01")
    fig.set_facecolor("#FFFFFF")
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    fig.savefig( settings.FIGURES_PAPER_DIR /  "FIG4.pdf")
    print(f'Figure 4 saved to: {settings.FIGURES_PAPER_DIR / 'FIG4.pdf'} ')


if False: # FIG 5 
    """ 
    Makes three plots 
    1) showing errors as a function of speed_error and the latitude 
    2) Showing the regression (predicted errors on the same data)
    """
    ##________________________________
    ## FIG5
    fig = plt.figure(figsize=(5,5), dpi = 400)
    gs = gridspec.GridSpec(3, 1)
    ax = fig.add_subplot(gs[-1])
    ax1 = fig.add_subplot(gs[:-1])
    cmap = plt.cm.inferno
    ofset = 6## ofsets in 4 hour incruments  # this is the start time if ofset = 6, first time plotted with 24hr. 
    timerange = 18
    variable = "initial_speed_dif_mag"
    for i in range(timerange):
        speedbins, binned_errors = opf.Projection_binning(merged,variable, i*2+ofset)
        ax1.scatter(speedbins[binned_errors.index.codes], binned_errors, label=f"{round((i*2+ofset)*4/24,1)}-{round((i*2+ofset+1)*4/24, 1)} days", color=cmap(i/timerange))

    nbins = 20
    counts, bin_edges = np.histogram(merged[variable].dropna(), bins=nbins)
    ax.bar(bin_edges[:-1], counts / counts.sum() * 100, width=np.diff(bin_edges), align='edge', alpha=0.7)
    ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set_ylabel("Frequency (%)")
    ax.set_xlabel(variable)
    ax.set_xlim(0,0.8)

    ax1.set_ylabel("Error km")
    ax1.set_xlabel(variable)
    ax1.set_xlim(0,0.8)
    ax1.set_ylim(0,150)
    ax1.set_title(f"{variable} vs \n displacement errors \n ")
    ax1.grid(alpha = 0.25)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    output_name  = plot_outputpath / 'FIG5.png'
    fig.savefig(output_name)
    print(f'Figure5 Saved to: {output_name}')

if False: #produces figure 6
    # Data setup for figures 6+ (binned error analysis)
    outputs = xr.load_dataset(output_data)

    angle_var = "initial_angle"
    speed_var = "initial_speed_dif_mag"
    lat_var  = "initial_lat"
    target_leadtime = 24*3
    target_quantile = 0.7

    merged = merged.sort_values(by = "leadtime")
    target_leadtime_lower = target_leadtime - 4
    merged_start = merged.query("leadtime > @target_leadtime_lower").drop_duplicates(subset = ["BuoyID", "starttime"])
    target_leadtime_upper = target_leadtime + 4
    merged_start = merged_start.query("leadtime < @target_leadtime_upper")

    # Create 2D bins
    speed_bins = np.linspace(0, 1, 8)
    lat_bins = np.arange(4.5,7.75,0.5)
    speed_bins = np.delete(speed_bins, [-3,-2])

    merged_start["speed_bin"] = pd.cut(merged_start[speed_var], speed_bins)
    merged_start["lat_bin"] = pd.cut(merged_start["initial_lat"], lat_bins)

    binned_data = merged_start.groupby(["lat_bin", "speed_bin"], observed=False)['error_km'].quantile(target_quantile).reset_index()
    lat_centers = merged_start.groupby(["lat_bin", "speed_bin"], observed=False)['initial_lat'].mean().reset_index()
    speed_center = merged_start.groupby(["lat_bin", "speed_bin"], observed=False)['initial_speed_dif_mag'].mean().reset_index()
    centers = pd.merge(lat_centers, speed_center, how = 'left')

    binned_pivot = binned_data.pivot_table(index="speed_bin", columns="lat_bin", values="error_km", observed = False)
    stacked = binned_pivot.stack().rename('error_km').reset_index(drop = False)
    # need to to get the larget intercelts from the regression
    q  = outputs.sel(leadtime = target_leadtime, q=target_quantile,  method = 'nearest')      

    centers['predicted_errors'] =q.Intercept.values + q.initial_speed_dif_mag.values*centers.initial_speed_dif_mag + q.initial_lat.values*centers.initial_lat
    centers_stacked = pd.merge(centers, stacked, on = [ 'lat_bin', 'speed_bin'], how = 'left')
    centers_stacked['error'] = centers_stacked.error_km - centers_stacked.predicted_errors
    centers_stacked[['lat_bin' , 'speed_bin', 'error']]
    centers_stacked['percent_error'] = np.abs(centers_stacked.error)/centers_stacked.predicted_errors
    centers_pivot = centers_stacked.pivot_table(index="speed_bin", columns="lat_bin", values="percent_error", observed = False, dropna= False)
    angle_mesh, speed_mesh = np.meshgrid(lat_bins, speed_bins)
    masked_data = np.ma.masked_invalid(centers_pivot.to_numpy())


    X,Y = np.meshgrid( lat_bins, speed_bins)
    error_values = q.Intercept.values  + q.initial_lat.values*X + q.initial_speed_dif_mag.values*Y
    fig, ax = plt.subplots(figsize = (6,6), dpi = 400)
    cbar = ax.contourf(X,Y,error_values, levels = 40, cmap='viridis', vmin=25, vmax=100)


    ax.pcolor(angle_mesh, speed_mesh, masked_data, facecolor='none', edgecolor='black', linewidth=1)

    to_label = centers_stacked.dropna(subset=["initial_lat", "initial_speed_dif_mag", "percent_error"])
    for _, row in to_label.iterrows():
        ax.text(
            row["initial_lat"],
            row["initial_speed_dif_mag"],
            f"{row['percent_error']*100:.1f}%",
            ha="center",
            va="center",
            fontsize=8,
            color="black",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.3),
        )
    #ax.contour(X,Y,error_values, levels = np.arange(25, 150, 25), colors = "k", ls = "--", alpha = 0.5)
    fig.colorbar(cbar, label = "Error (km)")
    ax.set(xlim = [4.5,7.5])
    ax.set_xlabel("Initial latitude")
    ax.set_ylabel("Initial Speed Difference")
    ax.set_title(f"Perdicted error Based off Inital Speed error and intial latitude\n Quantile: {target_quantile:.2f}\n at leadtime: {target_leadtime} \n  z = { q.Intercept:.2f} +{q.initial_lat:.3f}* lat + {q.initial_speed_dif_mag:.2f} * Speed error ", pad= 20)
    plt.subplots_adjust(top=0.8)
    output_name = plot_outputpath / 'FIG6.pdf'
    fig.savefig(output_name)
    print(f'Figure6 saved to: {output_name}')



    R_s = 1 - np.sum((centers_stacked.error_km - centers_stacked.predicted_errors)**2)/np.sum((centers_stacked.error_km - np.mean(centers_stacked.error_km))**2)
    print(f'Corrilation between regression qunatile and observed quantile : {R_s:.2f}')

if False:  ## produces figure 7
    import functions.draw_forecast_cones as cones
    fc0 = pd.read_csv(settings.FORECAST_DIR / 'cmems_bias_pers_meanremoved_2026.csv')
    dFADs = gpd.read_parquet(settings.dFAD_DATA)
    qdata = xr.load_dataset(settings.DATA_DIR / 'regression_quantiles_leadtimes_cmems_bias_pers_2026.nc')

    fc0.Time = pd.to_datetime(fc0.Time)

    merged_fc0 = opf.merged_dataframe_add_all_columns(fc0, dFADs)

    fig , axs = plt.subplots(1,2, figsize = (10,5), dpi = 400)
    ax0 , ax1 = axs.flatten()
    cones.plot_Forecast_from_dFAD_index([merged_fc0], dFADs, qdata, 255, 2, fig, ax0)
    cones.plot_Forecast_from_dFAD_index([merged_fc0], dFADs, qdata, 278, 1, fig, ax1)
    output_name = plot_outputpath / 'FIG7.pdf'
    fig.savefig(output_name)
    print(f'Figure7: saved to: {output_name}')

if False: # produces figure 8
    #_________________________________
    ## Figure 6: yearly variations 
    # cmems = xr.open_dataset(r'Data\cmems_monthly.nc')
    cmems = xr.open_dataset(settings.DATA_DIR / 'cmems_monthly.nc')
    cmems2022 = cmems.sel(time = slice("2022-01-01", "2023-01-01"), depth = 13.46714)
    cmems2023 = cmems.sel(time = slice("2023-01-01", "2024-01-01"), depth = 13.46714)
    cmems2024 = cmems.sel(time = slice("2024-01-01", "2025-01-01"), depth = 13.46714)
    cmems2025 = cmems.sel(time = slice("2025-01-01", "2026-01-01"), depth = 13.46714)

        ### recreating the figure above  to be in one time series with profiles for each year plotted below
    from matplotlib.gridspec import GridSpec
    # loading data
    merged["startday"] = merged.groupby(['BuoyID', 'starttime'], observed= False)['starttime'].transform('first')
    merged['startday'] = merged['startday'].dt.date
    bins = np.linspace(0,8*24,2*24+1)
    merged["lead_bins"] = pd.cut(merged["leadtime"], bins)
    binlist = merged["lead_bins"].unique()
    a  =binlist[7]
    fclt =  merged.groupby('lead_bins', observed= False).get_group(a).copy() 
    fclt_daily = fclt.groupby('startday', observed= False)['error_km'].mean()
    fclt_daily = fclt_daily.to_frame(name ='error_km').reset_index()
    fclt_daily['error30day'] = fclt_daily['error_km'].rolling(30,1,center = True).mean()

    fclt_daily['starttime'] = pd.to_datetime(fclt_daily.startday)
    fclt_daily['month'] = fclt_daily.starttime.dt.month
    fclt_daily['day'] = fclt_daily.starttime.dt.day


    upperlat = 10
    box = cmems2022.sel(latitude = slice(2, upperlat), longitude = slice(-163.75, -160.5))
    box = box.mean(dim = "longitude")
    box2 =cmems2023.sel(latitude = slice(2,upperlat), longitude = slice(-163.74, -160.5))
    box2 = box2.mean(dim = "longitude")
    box3 =cmems2024.sel(latitude = slice(2,upperlat), longitude = slice(-163.74, -160.5))
    box3 = box3.mean(dim = "longitude")
    box4 =cmems2025.sel(latitude = slice(2,upperlat), longitude = slice(-163.74, -160.5))
    box4 = box4.mean(dim = "longitude")

    #Calc Poofiles
    
    def calc_lat_average(longlist):
        profiles = pd.DataFrame()
        for n, bin in enumerate(longlist.season.unique().dropna().sort_values()):
            season = longlist[longlist['season'] == bin]
            profile =season.groupby('lat_bin', observed=False)['x_speed'].mean()
            profiles[n+1] = profile
        profiles = profiles.reset_index()
        profiles['lat'] = lat_bins[profiles.index] + np.diff(lat_bins)/2
        return profiles

    longlist['year'] = longlist.Time.dt.year
    longlist['month'] = longlist.Time.dt.month
    month_bins = np.array([1,4,7,10,13])
    longlist['season'] = pd.cut(longlist['month'], month_bins, right = False) # makes it [a,b) months 1-3, 4-6,7-9, 10-12
    lat_bins = np.arange(4.5,8.01, 0.50)
    longlist['lat_bin'] = pd.cut(longlist.lats, lat_bins, right = False)

    longlist2022 = longlist[longlist['year'] == 2022]
    longlist2023 = longlist[longlist['year'] == 2023]
    longlist2024 = longlist[longlist['year'] == 2024]
    longlist2025 = longlist[longlist['year'] == 2025]

    profiles2022 = calc_lat_average(longlist2022)
    profiles2023 = calc_lat_average(longlist2023)
    profiles2024 = calc_lat_average(longlist2024)
    profiles2025 = calc_lat_average(longlist2025)

    fig= plt.figure(figsize=(10,5), dpi=500)
    gs = GridSpec(2,4)
    ax0 = fig.add_subplot( gs[1,:])
    ax22 = fig.add_subplot(gs[0,0])
    ax23 = fig.add_subplot(gs[0,1])
    ax24 = fig.add_subplot(gs[0,2])
    ax25 = fig.add_subplot(gs[0,3])

    def calc_var(group):
        return group.x_speed.var() + group.y_speed.var()
    longlist['day'] = longlist.Time.dt.date
    varts = longlist.groupby('day', observed = False).apply(calc_var, include_groups=False).reset_index(name = 'var')
    varts = varts.rename(columns = {'day': 'startday'})
    varts['var30day'] = varts['var'].rolling(30,10, center = True).mean()
    varts['var90day'] = varts['var'].rolling(90,10, center = True).mean()
    ax0b = ax0.twinx()
    ax0.plot(varts.startday, varts.var30day , label = r'30 Day mean dFAD $\sigma^2$')
    ax0b.plot(fclt_daily.startday, fclt_daily.error30day, color = 'k', label = '30 day mean Forecast error')
    ax0.set(ylabel = r'$\sigma ^2$',xlim = [pd.Timestamp('2022-01-01'), pd.Timestamp('2026-01-01')])
    ax0.tick_params(labelrotation = 45)
    for d in pd.to_datetime(['2022-01-01', '2023-01-01', '2024-01-01', '2025-01-01']):
        ax0.axvline(d, color='k', lw=0.8)
    ax0b.set_ylabel('72hr Forecast error (km)')
    #x0.set_xticks(['2024'], )

    # Add corrilations to the plots. 

    # seasonal color strip on ax0 (acts as key for lower-panel colors)
    season_colors = ['r', 'g', 'b', 'purple']  # Jan-Mar, Apr-Jun, Jul-Sep, Oct-Dec
    xmin, xmax = pd.Timestamp('2022-01-01'), pd.Timestamp('2026-01-01')

    for yr in range(2022, 2026):
        season_edges = [
            (pd.Timestamp(f'{yr}-01-01'), pd.Timestamp(f'{yr}-04-01'), season_colors[0]),
            (pd.Timestamp(f'{yr}-04-01'), pd.Timestamp(f'{yr}-07-01'), season_colors[1]),
            (pd.Timestamp(f'{yr}-07-01'), pd.Timestamp(f'{yr}-10-01'), season_colors[2]),
            (pd.Timestamp(f'{yr}-10-01'), pd.Timestamp(f'{yr+1}-01-01'), season_colors[3]),
        ]
        for s, e, c in season_edges:
            s_clip, e_clip = max(s, xmin), min(e, xmax)
            if s_clip < e_clip:
                ax0.axvspan(s_clip, e_clip, ymin=0.00, ymax=0.035, color=c, alpha=0.9, ec='none')

    # optional thin border above the strip
    ax0.plot([xmin, xmax], [0.035, 0.035], transform=ax0.get_xaxis_transform(), color='k', lw=0.6)

    labels = ['Jan-Mar', 'Apr-Jun', 'Jul-Sep', 'Oct-Dec']
    colors = ['r', 'g', 'b', 'purple']
    for i in range(4): 
        ax22.plot(profiles2022[i+1], profiles2022.lat, label = labels[i], color = colors[i])
        ax23.plot(profiles2023[i+1], profiles2023.lat, color = colors[i])
        ax24.plot(profiles2024[i+1], profiles2024.lat, color = colors[i])
        ax25.plot(profiles2025[i+1], profiles2025.lat, color = colors[i])
        ax22.scatter(-0.3+np.random.random(1)*0.03,profiles2022.iloc[profiles2022[i+1].idxmax()].lat,
        color = colors[i], marker = "o", clip_on = False, zorder = 100)
        ax23.scatter(-0.3+np.random.random(1)*0.03,profiles2023.iloc[profiles2023[i+1].idxmax()].lat,
        color = colors[i], marker = "o", clip_on = False, zorder = 100)
        ax24.scatter(-0.3+np.random.random(1)*0.03,profiles2024.iloc[profiles2024[i+1].idxmax()].lat,
        color = colors[i], marker = "o", clip_on = False, zorder = 100)
        ax25.scatter(-0.3+np.random.random(1)*0.03,profiles2025.iloc[profiles2025[i+1].idxmax()].lat,
        color = colors[i], marker = "o", clip_on = False, zorder = 100)
        #CMEMS
        ax22.plot(box.uo[i*3:(i+1)*3,:].mean(dim = "time"), box.latitude, color = colors[i], alpha = 0.35, ls = ':') 
        ax23.plot(box2.uo[i*3:(i+1)*3,:].mean(dim = "time"), box2.latitude, color = colors[i], alpha = 0.5, ls = ':')
        ax24.plot(box3.uo[i*3:(i+1)*3,:].mean(dim = "time"), box2.latitude, color = colors[i], alpha = 0.5, ls = ':')
        ax25.plot(box4.uo[i*3:(i+1)*3,:].mean(dim = "time"), box2.latitude, color = colors[i], alpha = 0.5, ls = ':')

    ax22.set_ylabel('Latitude')

    yearlabels = ['2022', '2023', '2024', '2025']
    for i, axi in enumerate([ax22, ax23,ax24,ax25]):
        axi.set(xlim = [-0.3, 0.75], ylim = [2, 9], title= yearlabels[i])
        axi.tick_params(axis='both', labelsize=8)
        axi.hlines([7.75,4.5], xmin = -0.4, xmax = 7.5, color = 'k', alpha = 0.5, ls = '--')
        axi.set_xticks([-0.2,0,0.2,0.4, 0.6])

    # Remove gaps between lower subplots and make them share y-axis appearance
    for axi in [ax23, ax24, ax25]:
        axi.set_yticklabels([])
        axi.set_ylabel('')

    # Adjust the GridSpec to remove horizontal spacing between lower plots
    gs.update(wspace=0, hspace=0.3)
        # ax25.plot(profiles2025[i+1], profiles2025.lat)

    ax0.plot([], [], c='k', ls=':', label='climatology')
    # shared x-label for the top-row profile panels
    top_axes = [ax22, ax23, ax24, ax25]
    fig.suptitle(r'Yearly Variations in Currents and Errors')
    # Reserve bottom space for legend below ax0
    # fig.tight_layout(rect=[0, 0.1, 1, 1])
    ax0_y0 = ax0.get_position().y0  # bottom of ax0 in figure coords after tight_layout
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, ax0_y0 - 0.1),
            bbox_transform=fig.transFigure, fancybox=True, shadow=True, ncol=3)
    x_center = (min(axi.get_position().x0 for axi in top_axes) + max(axi.get_position().x1 for axi in top_axes)) / 2
    y_text = max(min(axi.get_position().y0 for axi in top_axes) - 0.04, 0.01)
    fig.text(x_center, y_text, r'Zonal velocity (m/s)', ha='center', va='top')
    output_name = plot_outputpath / 'FIG8.png'
    fig.savefig(output_name, bbox_inches='tight')
    print(f'Figure8 saved to: {output_name}')

    # calc corrilaton between Variance rolling mean and dFAD forecast errors
    fclt_varts = fclt_daily.merge(varts, how = 'left', on = 'startday')
    print(fclt_varts['error30day'].corr(fclt_varts['var30day'].shift(3)))