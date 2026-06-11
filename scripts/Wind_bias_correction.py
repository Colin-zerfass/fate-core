"""
Notebook to calcuated corrialtions of dFAD to the wind and bias Correction...
Barebones just get optimal corrilation between model dFADs, can

For more detailed analysis, plots and such see notebook Wind Anlysis.ipynb
solves the linear regression of the  U = a = U_currents*b + U_wind*c

"""
import functions.funcs as func
import pandas as pd 
import xarray as xr 
import geopandas as gpd
import numpy as np 
import functions.settings as settings 



def calc_R_anything(U, W):
    U = U - np.mean(U)
    W = W - np.mean(W)
    num = np.mean(np.conjugate(U)*W)
    a = np.mean(np.conjugate(U)*U)
    b = np.mean(np.conjugate(W)*W)
    return num/np.sqrt(a*b)

def Calc_Z(W, U):
    W = W - np.mean(W)
    U = U - np.mean(U)
    return np.mean(np.conj(W)*U)/np.mean(np.conj(W)*W)

def Regression(data, U= 'U', W = 'W', Uo = 'Uo'):
    U  = data[U]  - np.mean(data[U])
    W  = data[W]  - np.mean(data[W])
    Uo = data[Uo] - np.mean(data[Uo])
    A = np.vstack([Uo, W]).T
    B = U
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(A, B, rcond=None) #try ridge regression 
    return coefficients 


def print_coefficients(name, coefficients):
    print(f'{name}')
    print(f'  Ocean: {coefficients[0]:+.4f}  |  {np.abs(coefficients[0]):.3f} @ {np.angle(coefficients[0], deg=True):.1f} deg')
    print(f'  ERA5:  {coefficients[1]:+.4f}  |  {np.abs(coefficients[1]):.3f} @ {np.angle(coefficients[1], deg=True):.1f} deg')


def print_contributions(name, coefficients, Uo_mean, W_mean):
    ocean_contrib = np.abs(coefficients[0]) * Uo_mean
    wind_contrib  = np.abs(coefficients[1]) * W_mean
    total = ocean_contrib + wind_contrib
    print(f'  Currents: {ocean_contrib/total*100:.2f}%')
    print(f'  Wind:     {wind_contrib/total*100:.2f}%')


def regression_u(longlist, coefficients, Uo='Uo', W='W', suffix=None):
    # Coefficients were fit on anomalies, so apply to anomalies then add back mean(U)
    Uo_anom = longlist[Uo] - longlist[Uo].mean()
    W_anom  = longlist[W]  - longlist[W].mean()
    longlist['Ureg_' + suffix] = coefficients[0]*Uo_anom + coefficients[1]*W_anom + longlist['U'].mean()
    return longlist


if __name__ == '__main__':
    #Map model name -> ocean current column. Add/remove entries here to change what is processed.
    models = {
        'OSCAR':     'Uoscar',
        'CMEMS_1m':  'Uo_1',
        'CMEMS_5m':  'Uo_5',
        'GLORYs':    'Uo',
        'CMEMS_30m': 'Uo_30',
        'stokes' : 'Ustokes', 
        'GLORYs + stokes' : 'U_st' 
    }


    ds = gpd.read_parquet(settings.dFAD_DATA)
    longlist = func.generate_longlist(ds, extra_columns = ['mapped_u', 'mapped_u_oscar',
                                                            'mapped_v', 'mapped_v_oscar',
                                                            'mapped_u_winds', 'mapped_v_winds',
                                                            'mapped_u_1', 'mapped_v_1',
                                                            'mapped_u_5', 'mapped_v_5',
                                                            'mapped_u_30', 'mapped_v_30',
                                                            'mapped_u_stokes', 'mapped_v_stokes'])
    longlist['Time']   = pd.to_datetime(longlist.Time)
    longlist['U']      = longlist.x_speed       + 1j*longlist.y_speed
    longlist['W']      = longlist.mapped_u_winds + 1j*longlist.mapped_v_winds
    longlist['Uo']     = longlist.mapped_u       + 1j*longlist.mapped_v
    longlist['Uo_1']   = longlist.mapped_u_1     + 1j*longlist.mapped_v_1
    longlist['Uo_5']   = longlist.mapped_u_5     + 1j*longlist.mapped_v_5
    longlist['Uo_30']  = longlist.mapped_u_30    + 1j*longlist.mapped_v_30
    longlist['Uoscar'] = longlist.mapped_u_oscar + 1j*longlist.mapped_v_oscar
    longlist['Ustokes'] = longlist.mapped_u_stokes + 1j*longlist.mapped_v_stokes
    longlist['U_st'] = longlist.Ustokes + longlist.Uo_30

    #longlist = longlist[longlist.Time < pd.Timestamp('2025-07-01')]

    W_mean = np.mean(np.abs(longlist.W))
    # Map model name -> ocean current column. Add/remove entries here to change what is processed.
    if False:  # testing the GEOFLOW
        models = {
            'OSCAR':     'Uoscar',
            'GLORYs':    'Uo',
            'Geos': 'Ugeo'
        }


        dFADs = 'SAT_MI_FAD_cleanedspeeds_2026-01-01_geos_mapped.parquet'
        ds = gpd.read_parquet('Data/'+ dFADs)
        longlist = func.generate_longlist(ds, extra_columns = ['mapped_u', 'mapped_u_oscar',
                                                                'mapped_v', 'mapped_v_oscar',
                                                                'mapped_u_winds', 'mapped_v_winds',
                                                            'mapped_u_geos', 'mapped_v_geos'])
        longlist['Time']   = pd.to_datetime(longlist.Time)
        longlist['U']      = longlist.x_speed       + 1j*longlist.y_speed
        longlist['W']      = longlist.mapped_u_winds + 1j*longlist.mapped_v_winds
        longlist['Uo']     = longlist.mapped_u       + 1j*longlist.mapped_v
        longlist['Uoscar'] = longlist.mapped_u_oscar + 1j*longlist.mapped_v_oscar
        longlist['Ugeo'] = longlist.mapped_u_geos + 1j*longlist.mapped_v_geos
        longlist = longlist.dropna()
    #longlist = longlist[longlist.Time < pd.Timestamp('2025-07-01')]

    W_mean = np.mean(np.abs(longlist.W))
    print('solved the equation U = a*Currents + b*Wind\n')
    for name, uo_col in models.items():
        suffix = name.lower().replace(' ', '_')
        coef = Regression(longlist, Uo=uo_col)
        longlist = regression_u(longlist, coef, Uo=uo_col, suffix=suffix)

        print(f'--- {name} ---')
        print_coefficients(name, coef)
        print_contributions(name, coef, np.mean(np.abs(longlist[uo_col])), W_mean)
        R_raw = calc_R_anything(longlist.U, longlist[uo_col])
        R_reg = calc_R_anything(longlist.U, longlist['Ureg_' + suffix])
        print(f'  R (raw): |R|={np.abs(R_raw):.3f}, angle={np.angle(R_raw, deg=True):.1f} deg')
        print(f'  R (reg): |R|={np.abs(R_reg):.3f}, angle={np.angle(R_reg, deg=True):.1f} deg')

        Z = Calc_Z(longlist[uo_col], longlist.U)
        longlist['Uz_' + suffix] = Z * longlist[uo_col]
        R_z = calc_R_anything(longlist.U, longlist['Uz_' + suffix])
        print(f'  Z coef:  {Z:+.4f}  |  {np.abs(Z):.3f} @ {np.angle(Z, deg=True):.1f} deg')
        print(f'  R (Z):   |R|={np.abs(R_z):.3f}, angle={np.angle(R_z, deg=True):.1f} deg')
        print()
    print('WIND')
    r_wind = calc_R_anything(longlist.U, longlist.W)
    print(f'  R (raw): |R|={np.abs(r_wind):.3f}, angle={np.angle(r_wind, deg=True):.1f} deg')
    z_wind  = Calc_Z(longlist.W, longlist.U)
    print(f'  Z coef:  {z_wind:+.4f}  |  {np.abs(z_wind):.3f} @ {np.angle(z_wind, deg=True):.1f} deg')

    # calcuating what Uo_clim_mean and W_clim_mean should be. 
    GLORYs  = xr.open_dataset(settings.GLORYS_FILE)
    depths = [ 0.494,  5.0782,  13.4671, 29.4447]
    for d in depths:
        GLORYs_mean = (GLORYs.sel(time = slice('2021-01-01', '2025-12-31'), depth = d)
                        .mean(dim= ['time', 'latitude', 'longitude']))
        print(f'Uo_clim_mean at {d}')
        print(f'{GLORYs_mean.uo.values :.4f}, {GLORYs_mean.vo.values :.4f}')

    ERA5  = xr.open_dataset(settings.ERA5_FILE)
    ERA5_mean = (ERA5.sel(time = slice('2021-01-01', '2025-12-31'))
                    .mean(dim= ['time', 'latitude', 'longitude']))
    print('W_clim_mean')
    print(ERA5_mean.uo.values, ERA5_mean.vo.values)

    print('dFAD_mean')
    print(f"U_dfad_mean  = [{longlist.U.mean().real:.6f}, {longlist.U.mean().imag:.6f}]")