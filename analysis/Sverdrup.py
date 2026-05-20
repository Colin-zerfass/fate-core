import pandas as pd 
import xarray as xr
import numpy as np 
import matplotlib.pyplot as plt 

def Calc_sverdrup(windm):
    rho = 1.3 ## density of Air [kg/m^3]
    C_d = 1.25*1e-3  # wind stress coefient

    ## U_w = np.sqrt(windm.uo**2 + windm.vo**2) ## Chat GPT says to Tau = rho * C_d * U * |U| ???? 

    dlat = np.deg2rad(np.gradient(windm.lat.values))
    dlon = np.deg2rad(np.gradient(windm.lon.values))

    print(f"Lat: {len(dlat)} \nlon: {len(dlon )}")
    R = 6371e3 #radius of earth
    dy = R * dlat
    dx = R * dlon
    
    dlat_lon = np.cos(np.deg2rad(windm.lat.values)) ## how dlat varies with latitude 

    #Generate grid of dlat and dlon 
    dy_grid = np.repeat(dy.reshape(-1,1),len(dx), axis = 1)

    dx_grid = (np.repeat(dx.reshape(1,-1),len(dy), axis = 0).T *dlat_lon).T #transpose to match 
    print(f'dy gird {dy_grid.shape}')
    print(f'dy gird {dx_grid.shape}')

    windm['tau_u'] = rho*C_d*windm.uo**2 # [N/m^2]
    windm['tau_v'] = rho*C_d*windm.vo**2 # [N/m^2]

    windm['dy_tau_u']= windm.tau_u.differentiate(coord ='lat')/dy_grid # [N/m^3]
    windm['dx_tau_v']= windm.tau_v.differentiate(coord ='lon')/dx_grid # [N/m^3]
    windm['curl'] = windm.dx_tau_v - windm.dy_tau_u # [N/m^3]
    windm['dy_curl'] = windm.curl.differentiate(coord = 'lat')/dy_grid # [N/m^4]

    omega = 7.29e-5 # [1/s]
    lat = np.deg2rad(windm.lat) # [rad]

    B =  2*omega*np.cos(lat)/R # [1/m/s]

    rho_w = 1025 # [kg/m^3]
    # U = -(1/B)*windm['dy_curl'].sum(dim = 'lon')*dx_grid[:,0]/rho_w #[m^2/s]
    U = -(1/B)*windm['dy_curl'].integrate(coord = 'lon')*dx_grid[:,0]/rho_w
    return U
