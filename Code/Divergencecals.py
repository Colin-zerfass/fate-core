import xarray as xr 
import numpy as np 
import matplotlib.pyplot as plt 

import numpy as np
import scipy.sparse as sp

def build_operators_dirichlet_top(szY, szZ, dy, dz):
    import scipy.sparse as sp
    # A and B as in MATLAB
    A = np.column_stack([
    -1.0 * np.ones(szY),
    16.0 * np.ones(szY),
    -30.0 * np.ones(szY),
    16.0 * np.ones(szY),
    -1.0 * np.ones(szY),
    ])
    B = np.column_stack([
    np.ones(szY),
    -8.0 * np.ones(szY),
    np.zeros(szY),
    8.0 * np.ones(szY),
    -1.0 * np.ones(szY),
    ])
    K2y = (1.0 / (12.0 * dy**2)) * sp.spdiags(A.T, [-2, -1, 0, 1, 2], szY, szY)
    K1y = (1.0 / (12.0 * dy)) * sp.spdiags(B.T, [-2, -1, 0, 1, 2], szY, szY)
    Dy = sp.kron(sp.eye(szZ, format="csr"), K1y, format="csr")
    Dyy = sp.kron(sp.eye(szZ, format="csr"), K2y, format="csr")
    # z: B = fliplr(B);
    #Bz = np.fliplr(B) % no flip needed?
    Bz=B
    K1z = (1.0 / (12.0 * dz)) * sp.spdiags(Bz.T, [-2, -1, 0, 1, 2], szZ, szZ)
    K2z = (1.0 / (12.0 * dz**2)) * sp.spdiags(A.T, [-2, -1, 0, 1, 2], szZ, szZ)
    Dz = sp.kron(K1z, sp.eye(szY, format="csr"), format="csr")
    Dzz = sp.kron(K2z, sp.eye(szY, format="csr"), format="csr")
    return Dy, Dyy, Dz, Dzz
def stack2d(X):
    """Stack a 2-D array (nyxnx) into a 1-D column vector (ny*nx,)."""
    return X.ravel(order='C') # row-major flattening (consistent with your def unstack2d(v, ny, nx):

def unstack2d(v, ny, nx):
    """Unstack a 1-D vector into a 2-D array (nyxnx)."""
    return np.reshape(v, (ny, nx), order='C')

def padding(u:np.ndarray, w, nzeros): 
    u = np.pad(u,w, "linear_ramp", end_values= 0) 
    return np.pad(u, nzeros)

def unpad(u:np.ndarray, rows:int): 
    """Repoves the padding """
    return u[rows:-rows, rows:-rows]

def divergence(u:np.ndarray,v:np.ndarray,dx, dy, unstack = True): 
    """Calcs divergence field and absolute value"""
    n,m = u.shape 
    Dx,Dxx,Dy,Dyy = build_operators_dirichlet_top(n,m,dx,dx)
    dx_u = Dx@stack2d(u)
    dy_v = Dy@stack2d(v)
    if unstack == False:
        return dx_u+dy_v
    else: 
        return unstack2d(dx_u+dy_v, n,m)

def vorticity(u:np.ndarray,v:np.ndarray,dx,dy, unstack = True):
    n,m = u.shape 
    Dx,Dxx,Dy,Dyy = build_operators_dirichlet_top(n,m,dx,dx)
    dx_v = Dx@stack2d(v)
    dy_u = Dy@stack2d(u)
    if unstack == False:
        return dx_v - dy_u
    else: 
        return unstack2d(dx_v - dy_u, n,m)

def divergencefree(u:np.ndarray,v:np.ndarray,dx,dy):
    from scipy.sparse.linalg import spsolve
    n,m = u.shape
    vort = vorticity(u,v,dx,dy, unstack = False)
    Dx,Dxx,Dy,Dyy = build_operators_dirichlet_top(n,m,dx,dx)
    LG2 = Dxx +Dyy 
    psi = spsolve(LG2, -vort)
    u_df = Dy@psi
    v_df = -Dx@psi
    u_df = unstack2d(u_df,n,m)
    v_df = unstack2d(v_df,n,m)
    return u_df , v_df

def Curlfree(u:np.ndarray,v:np.ndarray,dx,dy):
    from scipy.sparse.linalg import spsolve
    n,m = u.shape
    Dx,Dxx,Dy,Dyy = build_operators_dirichlet_top(n,m,dx,dx)
    div = divergence(u,v,dx,dy,unstack= False)
    LG2 = Dxx +Dyy 
    pV = spsolve(LG2, div)
    u_cf = unstack2d(Dx@pV, n,m)
    v_cf = unstack2d(Dy@pV,n,m)
    return u_cf , v_cf 


if __name__ == "__main__": 
    ### reconstruct the field calc difference and then check divergence 
    import pandas as pd 
    from pykrige.ok import OrdinaryKriging
    from functions.funcs import * 

    def reproduce_field(field:xr.DataArray, date: pd.Timestamp, nsamples: int): 
        """repoduces a velocity field from randomly sampled values within the field"""
        samples = nsamples
        slicet = field.sel(time = date, depth = 15, method = "nearest")
        n,m = slicet.uo.shape
    
        latr = np.random.randint(0,n, samples)
        lonr = np.random.randint(0,m, samples)
        lat = slicet.latitude
        lon = slicet.longitude
        
        us = []
        vs = []
        lats = []
        lons = []
        for i in range(samples): 
            us.append(slicet.uo[latr[i],lonr[i]].item())
            vs.append(slicet.vo[latr[i],lonr[i]].item())
            lats.append(lat[latr[i]].item())
            lons.append(lon[lonr[i]].item())
        
        ds2 = pd.DataFrame({"lons": lons, "lats": lats, "us": us, "vs":vs})
        a = len(ds2)
        ds2 = ds2.drop_duplicates().reset_index(drop = True)
        if (len(ds2)- a) > 0:
            print(f" dropped # collumns: {len(ds2)- a}")

        x = OrdinaryKriging(
                    ds2.lons, ds2.lats, ds2.us,
                    variogram_model='spherical',
                    verbose=False,
                    enable_plotting=False
                )

        y = OrdinaryKriging(
                ds2.lons, ds2.lats, ds2.vs,
                variogram_model='spherical',
                verbose=False,
                enable_plotting=False
            )
        ##Need to set lat lons Size 
        z_predx, ss_x = x.execute('grid', lon, lat)
        z_predy, ss_y = y.execute('grid', lon, lat)
        return slicet, z_predx, z_predy, ss_x, ss_y
    
    def calc_statistics( slicet, z_predx , z_predy): 
        ru = np.mean(slicet.uo.values - z_predx)
        rv = np.mean(slicet.vo.values - z_predy)
        
        rmseu = np.sqrt(np.mean((slicet.uo.values - z_predx)**2))
        rmsev = np.sqrt(np.mean((slicet.vo.values - z_predy)**2))
        velomax = np.sqrt(z_predx**2 + z_predy**2).max()

        dx = haversine_dist(6.1, 163, 6.1, 163 +1/12) *1000
        cmems_div = divergence(slicet.uo.values, slicet.vo.values,dx,dx)
        krig_div = divergence(z_predx,z_predy, dx,dx)
        return ru, rv , rmseu, rmsev, cmems_div, krig_div, velomax

 ##need to test the date variablity in the conditions, variablity in the number of samples, Variability in the locations of samples
    cmems =xr.open_dataset("cmems.nc")
    print(len(cmems.latitude))
    print(len(cmems.longitude))

    if False : ##Resampling to check variablity in the locations of dFADs sampled. 
        nresamples = 200
        ds = pd.DataFrame(columns = ["ru", "rv" , "rmseu", "rmsev", "cmems_div", "krig_div", "velomax"])
    

        for i in range(nresamples):
            slicet, z_predx, z_predy, ss_x, ss_y = reproduce_field(cmems, pd.to_datetime("2024-09-27"), 40)
            row = calc_statistics(slicet, z_predx , z_predy)
            ds.loc[len(ds)] = row
        ds.to_csv(r"Data\krigging_test_stats.csv")

    if False: ## Testing the number of locations samples. 
        nresamples = np.linspace(2,200,(200-2)+1, dtype= int)
        print(nresamples[:5])
        ds = pd.DataFrame(columns = ["samples", "ru", "rv" , "rmseu", "rmsev", "cmems_div", "krig_div", "velomax"])
        for i in nresamples:
            slicet, z_predx, z_predy, ss_x, ss_y = reproduce_field(cmems, pd.to_datetime("2024-09-27"), i)
            row = [i]
            row.extend(calc_statistics(slicet, z_predx , z_predy))
            
            
            ds.loc[len(ds)] = row
        ds.to_csv(r"Data\krigging_test_stats_nsamples.csv")

    if True: 
        ds = pd.DataFrame(columns = ["ru", "rv" , "rmseu", "rmsev", "cmems_div", "krig_div", "velomax"])
        timerange = pd.date_range("2024-1-1", "2024-12-31")
        print(timerange[0:5])

        for day in timerange: 
            slicet, z_predx, z_predy, ss_x, ss_y = reproduce_field(cmems, day, 100)
            row = calc_statistics(slicet, z_predx , z_predy)
            ds.loc[len(ds)] = row
        ds.to_csv(r"Data\krigging_test_stats_days.csv")



    if False: ### Basic Testing of a time 
        ru = slicet.uo.values - z_predx
        rv = slicet.vo.values - z_predy
        print(f"Residuales of u: {np.mean(np.abs(ru))}")
        print(f"Residuales of v: {np.mean(np.abs(rv))}")
        rmseu = np.sqrt(np.mean((slicet.uo.values - z_predx)**2))
        rmsev = np.sqrt(np.mean((slicet.vo.values - z_predx)**2))
        print(f"RMSE u: {rmseu}")
        print(f"RMSE v: {rmsev}")
        dx = haversine_dist(6.1, 163, 6.1, 163 +1/12) *1000
        cmems_div = divergence(slicet.uo.values, slicet.vo.values,dx,dx)
        krig_div = divergence(z_predx,z_predy, dx,dx)
        print(f"cmems divergence magnitude: {np.mean(np.abs(cmems_div))}")
        print(f"krigging divergence magnitude: {np.mean(np.abs(krig_div))}")

    if True: 
        nresamples = 200
        ds = pd.DataFrame(columns = ["ru", "rv" , "rmseu", "rmsev", "cmems_div", "krig_div", "velomax"])
    
        cmems = []
        kriging = []
        for i in range(nresamples):
            slicet, z_predx, z_predy, ss_x, ss_y = reproduce_field(cmems, pd.to_datetime("2024-09-27"), 40)
            ru, rv , rmseu, rmsev, cmems_div, krig_div, velomax = calc_statistics(slicet, z_predx , z_predy)
            cmems_div = cmems_div.flatten()
            kriging_div = kriging_div.flatten()
            ds.loc[len(ds)] = row
        ds.to_csv(r"Data\krigging_test_stats.csv")
        


    



