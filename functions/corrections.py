"""
Shared statistical correction functions used by both Wind_bias_correction.py
and Dataloader_alligner.py (compute_bias_corrections).

Calc_Z   : optimal complex Z-scaling (single-predictor bias correction)
Regression : two-predictor complex least-squares (currents + wind)
"""
import numpy as np


def Calc_Z(W, U):
    """Optimal complex scalar Z such that U_anom ≈ Z * W_anom.
    W is the predictor (e.g. ocean current), U is the target (e.g. dFAD velocity).
    """
    W = W - np.mean(W)
    U = U - np.mean(U)
    return np.mean(np.conj(W) * U) / np.mean(np.conj(W) * W)


def Regression(data, U='U', W='W', Uo='Uo'):
    """Two-predictor complex lstsq on anomalies: U_anom = m*Uo_anom + n*W_anom.
    data : DataFrame with columns U, W, Uo (or as specified by keyword args).
    Returns complex coefficient array [m, n].
    """
    U_a  = data[U]  - np.mean(data[U])
    W_a  = data[W]  - np.mean(data[W])
    Uo_a = data[Uo] - np.mean(data[Uo])
    A = np.vstack([Uo_a, W_a]).T
    coefficients, _, _, _ = np.linalg.lstsq(A, U_a, rcond=None)
    return coefficients

def calc_R_anything(U, W):
    U = U - np.mean(U)
    W = W - np.mean(W)
    num = np.mean(np.conjugate(U)*W)
    a = np.mean(np.conjugate(U)*U)
    b = np.mean(np.conjugate(W)*W)
    return num/np.sqrt(a*b)

def regression_u(longlist, coefficients, Uo='Uo', W='W', suffix=None):
    # Coefficients were fit on anomalies, so apply to anomalies then add back mean(U)
    Uo_anom = longlist[Uo] - longlist[Uo].mean()
    W_anom  = longlist[W]  - longlist[W].mean()
    longlist['Ureg_' + suffix] = coefficients[0]*Uo_anom + coefficients[1]*W_anom + longlist['U'].mean()
    return longlist
