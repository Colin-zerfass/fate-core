import numpy as np 
import pandas as pd
import geopandas as gpd
import scipy.signal as signal

class Trajectory:
    def __init__(self, u, v, t):
        self.u = np.asarray(u, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.t = np.asarray(t, dtype='datetime64[ns]')

    @property
    def U(self):
        return self.u + 1j*self.v
    @property
    def Umean(self):
        return np.mean(self.U)
    @property
    def sigma2(self):
        return (np.mean((self.U.real - self.Umean.real)**2)
            + np.mean((self.U.imag - self.Umean.imag)**2))
    @property 
    def deltat(self):
        return np.diff(self.t)/np.timedelta64(1, 'h')
    @property
    def traj_length(self):
        return len(self.u)
    @property
    def dt(self):
        return self.t[1]-self.t[0]

    def normalize_t(self, dt:int):
        """interpolates trajectory onto even dt. 
        dt: [hours]"""
        dt64 = np.timedelta64(dt, 'h')
        t_new = np.arange(self.t[0], self.t[-1] + dt64, dt64)
        t_abs = self.t.astype(float)
        ti_abs = t_new.astype(float)
        u_new = np.interp(ti_abs, t_abs, self.u)
        v_new = np.interp(ti_abs, t_abs, self.v)
        out = Trajectory(u_new, v_new, t_new)
        return out

    def splice_dt_toolarge(self, maxdt=26):
        """
        Split trajectory where timestep exceeds maxdt hours.
        """
        split_idx = np.where(self.deltat > maxdt)[0] + 1
        t_split = np.split(self.t, split_idx)
        u_split = np.split(self.u, split_idx)
        v_split = np.split(self.v, split_idx)

        return [Trajectory(u, v, t)
            for u, v, t in zip(u_split, v_split, t_split)
            if len(t) > 1]
    
    def splice_even_trajectories(self, trajectory_length: int):
        """Split trajectory into equal-length chunks."""

        traj_length = np.timedelta64(trajectory_length, 'D')
        # skip trajectories that are too short
        dt_total = self.t[-1] - self.t[0]
        if dt_total < traj_length:
            return []

        # ensure normalize_t() was called
        if not hasattr(self, 'dt'):
            raise ValueError('trajectory must first be interpolated onto constant dt using normalize_t()')

        # trajectory_length must be multiple of dt
        multiple = traj_length / self.dt

        if not multiple.is_integer():
            raise ValueError('trajectory_length must be multiple of dt')

        step_idx = int(multiple)
        # number of complete chunks
        n_complete = len(self.t) // step_idx

        # truncate incomplete remainder
        cutoff = n_complete * step_idx
        t = self.t[:cutoff]
        u = self.u[:cutoff]
        v = self.v[:cutoff]
        split_idx = list(range(step_idx, cutoff, step_idx))
        t_split = np.split(t, split_idx)
        u_split = np.split(u, split_idx)
        v_split = np.split(v, split_idx)

        return [
            Trajectory(u_i, v_i, t_i)
            for u_i, v_i, t_i in zip(
                u_split,
                v_split,
                t_split)]
    
def dFAD_trajectry(dFADs: gpd.GeoDataFrame, index:int, u = 'x_speed', v = 'y_speed')->Trajectory:
    return Trajectory(dFADs.at[index, u], dFADs.at[index, v], dFADs.at[index, 'TimeStamp'])
class autocorrilation:
    def __init__(self, trajectory:Trajectory):
        # require regular interpolation
        if not hasattr(trajectory, "dt"):
            raise ValueError("trajectory must first be normalized with normalize_t()")

        self.traj = trajectory
        self.u = trajectory.u
        self.v = trajectory.v
        self.U = trajectory.U
        self.t = trajectory.t

        self.Umean = trajectory.Umean
        self.sigma2 = trajectory.sigma2
        self.dt = trajectory.dt
    @property
    def trajectory_length(self):
        return self.t[-1] - self.t[0]
    
    def calc_autocorrelation(self, tau):
        """Vector autocorrelation at lag tau (hours)."""
        tau = np.timedelta64(tau, 'h')
        taui = int(tau / self.dt)
        if taui == 0:
            return 1.0
        N = len(self.U)
        U_prime = self.U - self.Umean
        numerator = np.sum(
            U_prime[:-taui].real * U_prime[taui:].real
            + U_prime[:-taui].imag * U_prime[taui:].imag) / N
        return numerator / self.sigma2

    def calc_autocorrelation_series(self, tau):
        tau = np.timedelta64(tau, 'h')
        taui = int(tau / self.dt)
        seriesu = pd.Series(self.U.real)
        seriesv = pd.Series(self.U.imag)
        seriesspeed = pd.Series(np.abs(self.U))
        return (
            seriesu.autocorr(taui),
            seriesv.autocorr(taui),
            seriesspeed.autocorr(taui))
    
    def calc_autocorr_alltau(self, method='Vector'):
        Tau = []
        R = []
        dt_hours = int(self.dt / np.timedelta64(1, 'h'))
        for i in range(len(self.t) - 1):
            tau = i * dt_hours
            Tau.append(i * self.dt)
            if method == 'Vector':
                R.append(self.calc_autocorrelation(tau))

            elif method == 'Series':
                R.append(self.calc_autocorrelation_series(tau))
        return Tau, R    
    
def calc_autocorrilation(dFADs: gpd.GeoDataFrame, segment_length: int, Method='Vector', maxdt=48, dt = 4,  u = 'x_speed', v = 'y_speed') -> tuple[pd.DataFrame, list[int]]:
    """Calculates Autocorrilation on the entire dFAD dataset, producing a pandas dataframe of Tau, and Corrilations"""
    Tau_list = []
    R_list = []
    Ru_list = []
    Rv_list = []
    Rspeed_list = []
    traj_lengths = []
    traj_idx = []
    buoy_name_list = []
    segment_num_list = []
    startdate_list = []
    idx = 0

    for dFAD in range(len(dFADs)):
        dFAD_trajectry(dFADs, dFAD, u= u , v = v)

        traj = dFAD_trajectry(dFADs, dFAD, u= u , v = v)
        buoy_name = dFADs.at[dFAD, 'BuoyName']
        segments = traj.splice_dt_toolarge(maxdt=maxdt)
        buoy_chunk_idx = 0

        for seg in segments:
            if len(seg.t) <= 1:
                continue
            seg_norm = seg.normalize_t(dt=dt)
            chunks = seg_norm.splice_even_trajectories(segment_length)
            for chunk in chunks:
                acn = autocorrilation(chunk)
                traj_lengths.append(acn.trajectory_length)
                Tau, R = acn.calc_autocorr_alltau(method=Method)
                traj_idx.extend([idx] * len(Tau))
                buoy_name_list.extend([buoy_name] * len(Tau))
                segment_num_list.extend([buoy_chunk_idx] * len(Tau))
                chunk_sd = chunk.t[0]
                startdate_list.extend([chunk_sd]*len(Tau))
                buoy_chunk_idx += 1
                idx += 1
                Tau_list.extend(Tau)
                if Method == 'Vector':
                    R_list.extend(R)
                elif Method == 'Series':
                    Ru_list.extend([r[0] for r in R])
                    Rv_list.extend([r[1] for r in R])
                    Rspeed_list.extend([r[2] for r in R])

    if Method == 'Vector':
        return pd.DataFrame({"Tau": Tau_list, "R": R_list, "BuoyName": buoy_name_list, "segment_number": segment_num_list, 'starttime': startdate_list}, index=traj_idx), traj_lengths
    if Method == 'Series':
        return pd.DataFrame({'Tau': Tau_list, 'Ru': Ru_list, 'Rv': Rv_list, 'Rspeed': Rspeed_list}, index=traj_idx), traj_lengths

def autocorrelation_matrix(data: pd.DataFrame, value_column: str = 'R') -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Convert autocorrelation dataframe into matrix form."""
    # Convert long-form dataframe into matrix form
    pivot = data.pivot_table(index=data.index,columns='Tau',values=value_column)

    pivot = pivot.sort_index(axis=1)
    A = pivot.to_numpy()
    taus = pivot.columns.to_numpy()
    traj_ids = pivot.index.to_numpy()
    return A, taus, traj_ids

def matrix_to_acf_df( A: np.ndarray, Taus: np.ndarray, Traj_id: 
                     np.ndarray, value_column: str = 'R') -> pd.DataFrame:
    """Convert matrix-form autocorrelation data back into
    long-form pandas dataframe. """
    df = pd.DataFrame(A,index=Traj_id,columns=Taus)
    df.index.name = 'Trajectory'
    df = df.stack().reset_index()
    df.columns = ['Trajectory','Tau', value_column]
    df = df.set_index('Trajectory')
    return df

def bootstrap_autocorrilation(data, n_resamples):
    A, taus, traj_id = autocorrelation_matrix(data)
    """Bootstrap autocorrelation matrix"""
    ntraj = A.shape[0]
    # Bootstrap trajectory indices
    idx = np.random.randint(0,ntraj,size=(n_resamples, ntraj))
    # Mean autocorrelation for each bootstrap sample
    boot_means = A[idx].mean(axis=1)
    result = pd.DataFrame({
            'R': boot_means.mean(axis=0),
            'R_975': np.quantile(boot_means, 0.975, axis=0),
            'R_025': np.quantile(boot_means, 0.025, axis=0),},index=taus)
    result.index.name = 'Tau'
    return result

def block_bootstrap(data: pd.DataFrame, n_resamples: int, window_size: int = 20) -> pd.DataFrame:
    """
    Block bootstrap for autocorrelation data, where blocks are defined by
    contiguous time windows of `window_size` days based on the `starttime` column.

    Blocks are resampled with replacement. Each resample draws n_blocks blocks
    (same number as in the original data) and computes the mean ACF across all
    trajectories in the selected blocks.
    """
    A, taus, traj_ids = autocorrelation_matrix(data)  # (n_traj, n_tau)

    # One starttime per trajectory (index order matches A rows via traj_ids)
    meta = data[['starttime']].groupby(data.index).first()
    start_times = pd.to_datetime(meta.loc[traj_ids, 'starttime'])

    # Assign each trajectory to an integer block based on starttime
    t0 = start_times.min()
    days_from_start = (start_times - t0).dt.total_seconds() / 86400
    block_ids = (days_from_start // window_size).astype(int).to_numpy()

    unique_blocks = np.unique(block_ids)
    n_blocks = len(unique_blocks)

    # Map block_id -> row indices in A
    block_to_rows = {b: np.where(block_ids == b)[0] for b in unique_blocks}

    boot_means = np.empty((n_resamples, len(taus)))
    for s in range(n_resamples):
        # Resample blocks with replacement
        sampled_blocks = np.random.choice(unique_blocks, size=n_blocks, replace=True)
        # Collect all trajectory rows from sampled blocks
        row_idx = np.concatenate([block_to_rows[b] for b in sampled_blocks])
        boot_means[s] = A[row_idx].mean(axis=0)

    result = pd.DataFrame({
        'R':     boot_means.mean(axis=0),
        'R_975': np.quantile(boot_means, 0.975, axis=0),
        'R_025': np.quantile(boot_means, 0.025, axis=0),
    }, index=taus)
    result.index.name = 'Tau'
    return result

def interp_results(data, interval = 0.25):

    """interp mean ACF onto intervals of x hrs"""
    start, end  = data.index[0], data.index[-1]
    index_new  = pd.timedelta_range(start, end, freq = pd.Timedelta(interval, 'hours'))
    data_interp = data.reindex(data.index.union(index_new))
    data_interp = data_interp.interpolate(method = 'index')
    return data_interp.loc[index_new]
class Powerspectrum:
    def __init__(self, trajectory:Trajectory):
        self.traj = trajectory
        self.u = trajectory.u
        self.v = trajectory.v
        self.U = trajectory.U
        self.t = trajectory.t

        self.Umean = trajectory.Umean
        self.sigma2 = trajectory.sigma2
        self.dt = trajectory.dt
        self.dtseconds = self.dt /np.timedelta64(1, 's')
        self.Uprime = self.U - self.Umean
        self.freq = 1/self.dtseconds
    
    def welch(self,data, window = 'hann_periodic'):
        f, PSD = signal.welch(data, 1/self.dtseconds, window = window)
        return f, PSD

    def calc_fft(self, data=None):
        if data is None:
            data = self.Uprime.real

        fft = np.fft.fft(data)
        fftfreq = np.fft.fftfreq(len(data), d=self.dtseconds)
        # positive_index = fftfreq > 0
        # fftfreq = fftfreq[positive_index]
        # fft = fft[positive_index]
        return fft, fftfreq,
    
    def calc_psd(self, fft): 
        psd = (2 /(self.freq*len(self.Uprime)))*  np.abs(fft)**2
        return psd
        
def calc_powerspectrum(dFADs: gpd.GeoDataFrame, segment_length: int, interp_dt = 4 , 
                       maxdt = 48, method = 'welch', window = 'hann_periodic', dimention = 'u', u = 'x_speed' , v = 'y_speed')-> tuple[pd.DataFrame, list[int]]:
    freqs= []
    psds = []
    trajidx = []
    VARS = []
    idx = 0
    for dFAD in range(len(dFADs)):
        traj = dFAD_trajectry(dFADs, index= dFAD, u = u , v =v )
        segments = traj.splice_dt_toolarge(maxdt=maxdt)
        for seg in segments:
            if len(seg.t) <= 1:
                continue
            seg_norm = seg.normalize_t(dt=interp_dt)
            chunks = seg_norm.splice_even_trajectories(segment_length)
            for chunk in chunks:
                ps = Powerspectrum(chunk)
                data = ps.Uprime.real if dimention == 'u' else ps.Uprime.imag
                if method == 'welch':
                    f, psd = ps.welch(window = window, data = data)
                if method == 'fft':
                    fft, freq = ps.calc_fft(data)
                    psd = ps.calc_psd(fft)
                    mask = freq > 0 
                    f = freq[mask]
                    psd = psd[mask]
                freqs.extend(f)
                psds.extend(psd)
                trajidx.extend([idx]*len(f))
                VARS.extend([data.var()]*len(f))
                #print(np.trapezoid(psd,f), data.var())
                idx += 1
            
    return pd.DataFrame({'freq': freqs, 'PSD': psds, 'VARS': VARS}, index = trajidx)