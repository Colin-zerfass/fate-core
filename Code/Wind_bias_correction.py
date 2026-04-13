"""
Notebook to calcuated corrialtions of dFAD to the wind and bias Correction...
Barebones just get optimal corrilation between model dFADs, can

For more detailed analysis, plots and such see notebook Wind Anlysis.ipynb

"""
import functions.funcs as func
import pandas as pd 
import xarray as xr 
import geopandas as gpd
import numpy as np 



def calc_R_anything(U, W):
    num = np.mean(np.conjugate(U)*W)
    a = np.mean(np.conjugate(U)*U)
    b = np.mean(np.conjugate(W)*W)
    return num/np.sqrt(a*b)

def Regression(data, U= 'U', W = 'W', Uo = 'Uo'):
    U = data[U]
    W = data[W]
    Uo = data[Uo]
    A = np.vstack([Uo, W, np.ones(len(Uo))]).T
    B = U
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, B, rcond=None) #try ridge regression 
    return coefficients 


if __name__ == '__main__':
    
    dFADs = 'SAT_MI_FAD_cleanedspeeds_2026-01-01_mapped_all.parquet'
    ds = gpd.read_parquet('Data/'+ dFADs)
    longlist = func.generate_longlist(ds, extra_columns = ['mapped_u', 'mapped_u_oscar',
                                                            'mapped_v', 'mapped_v_oscar', 
                                                            'mapped_u_winds', 'mapped_v_winds'])
    longlist['U'] = longlist.x_speed + 1j*longlist.y_speed
    longlist['W'] = longlist.mapped_u_winds + 1j*longlist.mapped_v_winds
    longlist['Uo'] = longlist.mapped_u + 1j*longlist.mapped_v
    longlist['Uoscar'] = longlist.mapped_u_oscar + 1j*longlist.mapped_v_oscar

    coefficients_cmems = Regression(longlist)
    coefficients_oscar = Regression(longlist, Uo = 'Uoscar')

    print('the optimal use wind and current')
    print('solved the rquation of a + b*Currents +c*Wind \n ')

    print('GLORYs')
    print(coefficients_cmems)
    print(f'cmems: {np.abs(coefficients_cmems[0]):.2f}, {np.angle(coefficients_cmems[0],deg= True):.2f} degrees ')
    print(f'ERA5: {np.abs(coefficients_cmems[1]):.2f}, {np.angle(coefficients_cmems[1],deg= True):.2f} degrees\n')
    
    print('OSCAR')
    print(coefficients_oscar)
    print(f'OSCAR: {np.abs(coefficients_oscar[0]):.2f}, {np.angle(coefficients_oscar[0],deg= True):.2f} degrees ')
    print(f'ERA5: {np.abs(coefficients_oscar[1]):.2f}, {np.angle(coefficients_oscar[1],deg= True):.2f} degrees')


