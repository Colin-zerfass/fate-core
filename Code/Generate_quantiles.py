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


if plotting == True: 
    """ 
    Makes two plots 
    1) showing errors as a function of speed_error and the latitude 
    2) Showing the regression (predicted errors on the same data)
    """
    import matplotlib.pyplot as plt 
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
    print(lat_bins, speed_bins)
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
    print(error_values.shape)
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
    fig.savefig(plot_outputpath + "Forecast_errors_VS_lat_speed.pdf")

##____________________
    
    # need to to get the larget intercelts from the regression
    q  = outputs.sel(leadtime = target_leadtime, q=target_quantile,  method = 'nearest')      

    centers['predicted_errors'] =q.Intercept.values + q.initial_speed_dif_mag.values*centers.initial_speed_dif_mag + q.initial_lat.values*centers.initial_lat
    centers_stacked = pd.merge(centers, stacked, on = [ 'lat_bin', 'speed_bin'], how = 'left')
    centers_stacked['error'] = centers_stacked.error_km - centers_stacked.predicted_errors
    centers_stacked[['lat_bin' , 'speed_bin', 'error']]
    centers_stacked['percent_error'] = np.abs(centers_stacked.error)/centers_stacked.predicted_errors
    centers_pivot = centers_stacked.pivot_table(index="speed_bin", columns="lat_bin", values="percent_error", observed = False, dropna= False)
    print(centers_pivot.to_numpy().shape)
    angle_mesh, speed_mesh = np.meshgrid(lat_bins, speed_bins)
    masked_data = np.ma.masked_invalid(centers_pivot.to_numpy())


    X,Y = np.meshgrid( lat_bins, speed_bins)
    error_values = q.Intercept.values  + q.initial_lat.values*X + q.initial_speed_dif_mag.values*Y
    fig, ax = plt.subplots(figsize = (6,6), dpi = 400)
    cbar = ax.contourf(X,Y,error_values, levels = 100, cmap='viridis', vmin=25, vmax=100)

    ax.pcolor(angle_mesh, speed_mesh, masked_data, facecolor='none', edgecolor='black', linewidth=1)

    to_label = centers_stacked.dropna(subset=["initial_lat", "initial_speed_dif_mag", "percent_error"])
    print(to_label.shape)
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
    ax.set_title(f"Perdicted error Based off Inital Speed error and intial latitude\n Quantile: {target_quantile:.2f}\n at leadtime: {target_leadtime} \n  z = { q.Intercept:.2f} +{q.initial_lat:.3f}* lat + {q.initial_speed_dif_mag:.2f} * Speed error ")
    fig.savefig(plot_outputpath + "Paper/FIG5.pdf")




    R_s = 1 - np.sum((centers_stacked.error_km - centers_stacked.predicted_errors)**2)/np.sum((centers_stacked.error_km - np.mean(centers_stacked.error_km))**2)
    print(R_s)