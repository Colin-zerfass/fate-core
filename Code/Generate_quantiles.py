import numpy as np 
import pandas as pd 
import geopandas as gpd
import functions.funcs as funcs 
import functions.output_functions as opf
import xarray as xr
import os

"""Genrates Regression for forecast errors. Error(qauntile, leadtime, initial_speed_dif, latitude)

"""
dFAD_data  = r"Data\SAT_MI_FAD_cleanedspeeds_2026-01-01_mapped_all.parquet"
Forecast_data = r"Parcels\saved_output\Final\cmems_bias_pers_meanremoved_2026.csv"

output_data = r'Data\regression_quantiles_leadtimes_cmems_bias_pers_2026.nc'

calc_quantiles = False ## can turn to false if output file already exists and just want plots, reccomended takes about a minute to recalc this quantile 
plotting = True #to plot the pregression and reproduce the figure 5? 
plot_outputpath = r"../Figures/"
## True dFAD data 




ds = gpd.read_parquet(dFAD_data) 
fc = pd.read_csv(Forecast_data)

fc["Time"] = pd.to_datetime(fc["Time"])
fc['error_km'] = funcs.haversine_df(fc, "lat_true", "lon_true", "lat_forcast", "lon_forcast")

## Unpacking True dFAD data into one list
longlist = pd.DataFrame({})
longlist["Time"] = funcs.Column_to_List(ds, "TimeStamp", idlist = False)
longlist["lats"], longlist["lons"] = funcs.list_of_latlon(ds, False)
longlist["x_speed"] = funcs.Column_to_List(ds, "x_speed", idlist = False)
longlist["y_speed"] = funcs.Column_to_List(ds, "y_speed", idlist = False)
longlist["v_mapped"], longlist["BuoyID"]  =funcs.Column_to_List(ds, "mapped_v", idlist = True)
longlist["v_mapped_OSCAR"], longlist["BuoyID"]  =funcs.Column_to_List(ds, "mapped_v_oscar", idlist = True)
longlist["u_mapped"] = funcs.Column_to_List(ds, "mapped_u", idlist = False)
longlist["u_mapped_OSCAR"] = funcs.Column_to_List(ds, "mapped_u_oscar", idlist = False)
longlist.Time = pd.to_datetime(longlist.Time)

merged = opf.merge_forecast_true(fc, longlist)
merged = opf.add_starttime(merged)
merged["speed"] = np.sqrt(merged.x_speed**2 + merged.y_speed**2)
merged = opf.calc_intial_speed_dif(merged)
merged = opf.calc_iniial_lat(merged)


if calc_quantiles or not os.path.exists(output_data):

    output = opf.quantile_regression(merged)
    output.to_netcdf(output_data)


if plotting == True: # produces figure 4
    import matplotlib.pyplot as plt 
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import PercentFormatter
    if False:
        """ 
        Makes three plots 
        1) showing errors as a function of speed_error and the latitude 
        2) Showing the regression (predicted errors on the same data)
        """
        ##________________________________
        ## FIG4
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
        output_name  = r'Paper/FIG4.pdf'
        fig.savefig(plot_outputpath + output_name)
        print('Figure4 Saved to: ' + plot_outputpath +output_name)
    #________________________________________________________
    if False: 
        # reloads the output data
        outputs = xr.load_dataset(output_data)

        angle_var = "initial_angle"
        speed_var = "initial_speed_dif_mag"
        lat_var  = "initial_lat"
        plot_variables = ["initial_lat", "initial_speed_dif_mag"]
        target_leadtime = 24*3
        target_quantile = 0.7

        merged = merged.sort_values(by = "leadtime")
        target_leadtime_lower = target_leadtime - 4
        merged_start = merged.query("leadtime > @target_leadtime_lower").drop_duplicates(subset = ["BuoyID", "starttime"])
        target_leadtime_upper = target_leadtime + 4
        merged_start = merged_start.query("leadtime < @target_leadtime_upper") 


        # Create 2D bins for initial_angle and initial_speed_dif_mag
        angle_bins = np.linspace(0, 180, 8)
        speed_bins = np.linspace(0, 1, 8)
        lat_bins = np.arange(4.5,7.75,0.5)
        speed_bins = np.delete(speed_bins, [-3,-2])
        # Bin the data

        merged_start["speed_bin"] = pd.cut(merged_start[speed_var], speed_bins)
        merged_start["lat_bin"] = pd.cut(merged_start["initial_lat"], lat_bins)
        # Calculate mean error for each 2D bin  
        binned_data = merged_start.groupby(["lat_bin", "speed_bin"], observed=False)['error_km'].quantile(target_quantile).reset_index() #Calc Quantile 
        lat_centers = merged_start.groupby(["lat_bin", "speed_bin"], observed=False)['initial_lat'].mean().reset_index() # Calc, centers of each bin 
        speed_center = merged_start.groupby(["lat_bin", "speed_bin"], observed=False)['initial_speed_dif_mag'].mean().reset_index() 
        centers = pd.merge(lat_centers, speed_center, how = 'left')

        binned_pivot = binned_data.pivot_table(index="speed_bin", columns="lat_bin", values="error_km", observed = False)
        stacked = binned_pivot.stack().rename('error_km').reset_index(drop = False)
        # Create meshgrid for pcolormesh

        angle_mesh, speed_mesh = np.meshgrid(lat_bins, speed_bins)

        # Convert binned_pivot to 2D array for pcolormesh
        error_values = binned_pivot.values
        fig, ax = plt.subplots(figsize=(10, 6))
        pcm = ax.pcolormesh(angle_mesh, speed_mesh, error_values, cmap='viridis', vmin=25, vmax=100)
        cbar = fig.colorbar(pcm, ax=ax, label="Mean Error (km)")
        #pcm.set_clim(10,120)

        counts = merged_start.groupby(["lat_bin", "speed_bin"], observed=False).size().reset_index(name="count")
        counts_pivot = counts.pivot(index="speed_bin", columns="lat_bin", values="count")
        count_values = counts_pivot.values
        angle_centers = (lat_bins[:-1] + lat_bins[1:]) / 2
        speed_centers = (speed_bins[:-1] + speed_bins[1:]) / 2
        angle_cent_mesh, speed_cent_mesh = np.meshgrid(angle_centers, speed_centers)
        for (i, j), val in np.ndenumerate(count_values):
            cnt = count_values[i, j]
            if np.isnan(cnt):
                continue
            x = angle_cent_mesh[i, j]
            y = speed_cent_mesh[i, j]
            err = error_values[i, j] if not np.isnan(error_values[i, j]) else 0
            #text_color = "white" if err > (pcm.get_clim()[1] * 0.45) else "black"
            ax.text(x, y, int(cnt), ha="center", va="center", color="white", fontsize=8)
        # <-- END INSERTION

        ax.set_xlabel("Initial Latitude")
        ax.set_ylabel("Initial Speed Difference (m/s)")
        ax.set_title(f"0.7 Quantile vs latitude and Speed Difference \n Number are amount points in each bin \n Errors at leadtime: {target_leadtime +2} hrs")
        output_name = "Forecast_errors_VS_lat_speed.pdf"
        fig.savefig(plot_outputpath + output_name)
        print('Figure saved too: ' + plot_outputpath + output_name )
##____________________
    if False: #produces figure 5 
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
        cbar.set_rasterized(True)


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
        output_name = "Paper/FIG5.pdf"
        fig.savefig(plot_outputpath + output_name)
        print('Figure5 saved to: ' + plot_outputpath + output_name)



        R_s = 1 - np.sum((centers_stacked.error_km - centers_stacked.predicted_errors)**2)/np.sum((centers_stacked.error_km - np.mean(centers_stacked.error_km))**2)
        print(f'Corrilation between regression qunatile and observed quantile : {R_s:.2f}')

    if True: # produces figure 6
        #_________________________________
        ## Figure 6: yearly variations 
        # cmems = xr.open_dataset(r'Data\cmems_monthly.nc')
        cmems = xr.open_dataset(r'Data\cmems_monthly.nc')
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
        output_name = "Paper/FIG6.pdf"
        fig.savefig(plot_outputpath + output_name, bbox_inches='tight')
        print('Figure6 saved to: ' + plot_outputpath + output_name)
