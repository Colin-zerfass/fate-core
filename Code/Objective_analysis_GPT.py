# %%
import pandas as pd 
import geopandas as gpd
import numpy as np
from functions.funcs import *

# Notebook to test objective mapping using Bretherton et al 1975# %%
def C_psi(r, L, sigma2):
    return sigma2 * np.exp( -r/L)

def R_and_S(r, L, sigma2):
    F = C_psi(r, L, sigma2)
    dFdr = -(1/L)*F
    d2Fdr2 = (1/L**2)*F

    R = -(1/r)*dFdr      # R(ρ)
    S = -d2Fdr2          # S(ρ)
    return R, S

# Build full observation covariance A (2N x 2N) for stacked [u; v]
def obs_covariance(lons:np.array, lats:np.array, L, sigma_u_obs = 0.0002, sigma2_psi =1):

    N = len(lons) ##number of measurments 
    A = np.zeros((2*N, 2*N)) #Matrix to hold variences. 
    lons2 = np.append(lons, lons)
    lats2 = np.append(lats,lats)

    for i in range(N):
        xi, yi = lons2[i], lats2[i]
        for j in range(N):
            if i ==j: 
                A[i,j] = 1
                A[i+N, j+N] = 1
            else: 
                xj, yj = lons2[j], lats2[j]
                dx = xi - xj
                dy = yi - yj
                r= np.abs(haversine_dist(yj,yi, xj, xi))
                C = C_psi(r, L, sigma2_psi)
                u1u1 = (dx**2/r**2)*(sigma2_psi*C/L/r + sigma2_psi*C/L*2) - (sigma2_psi*C/L**2) ##Fix there are negetives here
                u2u2 = (dy**2/r**2)*(sigma2_psi*C/L/r + sigma2_psi*C/L*2) - (sigma2_psi*C/L**2) #and here
                u1u2 = (dx*dy/r**2)*(sigma2_psi*C/L/r + sigma2_psi*C/L*2) # actually negetives here

                A[i, j] = u1u1
                A[i+N, j] = u1u2
                A[i, j+N] = u1u2
                A[i+N, j+N] = u2u2

    # add observation noise variances on diagonal so inverse is solveable
    # for i in range(2*N):
    #     A[i, i] += sigma_u_obs**2
    return A

def phi_obs(u, v):
    return np.append(u,v)

def C_iu(lat_poi, lon_poi, lats, lons, L, sigma2_psi):
    N = len(lats)
    c = np.zeros(N*2)
    for i in range(N):
        print(lons[i])
        dx = lon_poi - lons[i]
        dy = lat_poi - lats[i]
        r = np.sqrt(dx**2 +dy**2)
        f = C_psi(r, L, sigma2_psi)
        u1u1 = (dx**2/r**2)*(sigma2_psi*f/L/r + sigma2_psi*f/L*2) - (sigma2_psi*f/L**2) ##Fix there are negetives here
        dx = lon_poi - lons[i+N]
        dy = lat_poi - lats[i+N]
        r = np.sqrt(dx**2 +dy**2)
        f = C_psi(r, L, sigma2_psi)
        u1u2 = (dx*dy/r**2)*(sigma2_psi*f/L/r + sigma2_psi*f/L*2) ##Fix there are negetives here
        c[i] = u1u1
        c[i+N] = u1u2
    return 

def U_xy(lat_poi, lon_poi, lats, lons, L, u,v, sigma2_psi):
    A = obs_covariance(lats, lons, L)
    Ai = np.linalg.inv(A)
    phi = phi_obs(u,v)
    cs = C_iu(lat_poi, lon_poi,lats, lons,L ,sigma2_psi )
    N = len(lats)
    b= 0
    for r in range(2*N):
        b += cs[r]*(Ai[r,:]*phi).sum()
    return b.sum()

def u_field(xrange, yrange, lats, lons, u,v,L, sigma2_psi):
    L = np.abs(haversine_dist(L,0,0,0))
    print(L)
    f = np.zeros((len(xrange), len(yrange)))
    for i in range(len(xrange)):
        print(i)
        for j in range(len(yrange)):
            f[i,j] = U_xy(xrange[i], yrange[j], lats, lons, L, u, v,sigma2_psi)
    return f

# %%
data = xr.open_dataset(r"data\cmems.nc")
df = ds_locations #: location data of dFADs for one timestep
u = u_field(data.longitude.to_numpy(), data.latitude.to_numpy(), df.lat, df.lon, df.x_speed, df.y_speed, L = 0.5, sigma2_psi = 1)