import numpy as np 
import pandas as pd
import geopandas as gpd




class Autocorrilation:
    def __init__(self, u,v,t ): 
        self.u = u #ds.at[i, "x_speed"]
        self.v = v #ds.at[i, "y_speed"]
        self.U = self.u+1j*self.v 
        self.t = t # ds.at[i, "TimeStamp"]
        self.t = np.array(self.t, dtype='datetime64')
        self.Umean = np.mean(self.U)
        self.sigma2 =np.mean((self.U.real - self.Umean.real)**2) + np.mean((self.U.imag - self.Umean.imag)**2)
        self.split = False
        return None
    def Normalize_t(self, dt:int):
        t_abs = self.t.astype("float64")/8.64e4
        self.dt = np.timedelta64(dt, 'h')
        self.t = np.arange(self.t[0], self.t[-1], self.dt, dtype='datetime64')
        ti_abs = self.t.astype("float64")/8.64e4
        self.u = np.interp(ti_abs,t_abs, self.u)
        self.v = np.interp(ti_abs,t_abs, self.v)
        self.U = self.u +1j*self.v
        self.Umean = np.mean(self.U)
        self.sigma2 = np.mean((self.U.real - self.Umean.real)**2) + np.mean((self.U.imag - self.Umean.imag)**2)
        return None
    
    def splice_even_trajectories(self, trajectory_length:int):
        # trajectory is already intperploated onto a constant dt
        self.traj_length = np.timedelta64(trajectory_length, 'D')
        #first skip trajectories that are too short
        self.dt_total = self.t[-1] - self.t[0]
        if self.dt_total < self.traj_length:
            self.tooshort = True 
            return
        else:
            self.tooshort = False
        # trajectory_length must be a multiple of dt 
        mutliple = self.traj_length/self.dt
        if mutliple.is_integer() == False:
            raise('not evenly spaced trajectory interpolate before using function or trajectory_length is not a multiple of dt')
        
        step_idx = int(self.traj_length / self.dt)  # number of timesteps per trajectory_length
        n_complete = len(self.t) // step_idx  # number of complete trajectories
        cutoff = n_complete * step_idx  # index to truncate at
        self.t = self.t[:cutoff]
        self.u = self.u[:cutoff]
        self.v = self.v[:cutoff]
        self.U = self.U[:cutoff]
        split_idx = list(range(step_idx, cutoff, step_idx))
        self.t = np.split(self.t, split_idx)
        self.u = np.split(self.u, split_idx)
        self.v = np.split(self.v, split_idx)
        self.U = np.split(self.U, split_idx)
        self.Umean = []
        self.sigma2 = []
        for i in range(len(self.U)):
            self.Umean.append(np.mean(self.U[i]))
            self.sigma2.append(np.mean((self.U[i].real - self.Umean[i].real)**2) + np.mean((self.U[i].imag - self.Umean[i].imag)**2))
        return

    def trajectory_length(self):
        self.traj_length = self.t[-1] - self.t[0]
        return None
    
    def calc_autocorrelation(self, tau):
        tau = np.timedelta64(tau, 'h')
        t = self.t
        dt = t[1]-t[0]
        taui = int(tau/dt)
        if taui == 0:
            return 1.0  # R(0) = 1 by definition; U_prime[:-0] is empty so handle explicitly
        N = len(self.U)
        ## Equation (3) Pasquero et al. 2007, biased estimator (divide by N)
        U_prime = self.U - self.Umean
        numerator = np.sum(U_prime[:-taui].real*U_prime[taui:].real + U_prime[:-taui].imag*U_prime[taui:].imag) / N
        return numerator/self.sigma2
    
    def calc_autocorrelation_series(self, tau):
        tau = np.timedelta64(tau, 'h')
        t = self.t ## check 1e9
        dt = t[1]-t[0]
        taui= int(tau/dt)
        seriesu = pd.Series(self.U.real)
        seriesv = pd.Series(self.U.imag)
        seriesspeed = pd.Series(np.abs(self.U))
        return seriesu.autocorr(taui) , seriesv.autocorr(taui)  , seriesspeed.autocorr(taui) 
    
    def calc_autocor_allTau(self, method ='Vector'):
        """Need to call Calc_autocorrelation for all Tau, First need to calc max range of Tau, then store Tau"""
        dtint = int(self.dt/np.timedelta64(1, "h"))
        self.Tau = []
        self.R = []
        self.Ru = []
        self.Rv = []
        self.Rspeed = []
        for i in range(len(self.t)-1):  # starts at i=0 (tau=0), up to len-2 (last valid lag)
            if method == 'Vector':
                r = self.calc_autocorrelation(i*dtint)
                self.R.append(r)
                self.Tau.append(i*self.dt)
            if method == 'Series':
                ru, rv, rspeed = self.calc_autocorrelation_series(i*dtint)
                self.Ru.append(ru)
                self.Rv.append(rv)
                self.Rspeed.append(rspeed)
                self.Tau.append(i*self.dt)
        return None    
    

class split_trajectory:
    def __init__(self, u,v,t): 
        self.u = u
        self.v = v
        self.U = self.u+1j*self.v
        self.t = t
        self.t = np.array(self.t, dtype='datetime64')
        self.split = False
        return None
    def splice_dt_toolarge(self,maxdt = 26):
        self.deltat = np.diff(self.t)/np.timedelta64(1, 'h')  ## converts to hours
        split_idx = np.where(self.deltat > maxdt)[0]
        self.t = np.split(self.t, split_idx+1)
        self.u = np.split(self.u, split_idx+1)
        self.v = np.split(self.v, split_idx+1)
        self.U = np.split(self.U, split_idx+1)
        self.Umean = []
        self.sigma2 = []
        for i in range(len(self.U)):
            self.Umean.append(np.mean(self.U[i]))
            self.sigma2.append(np.mean((np.abs(self.U[i] - self.Umean[i]))**2))
        self.split = True
        return None
    

def calc_autocorrilation(dFADs:gpd.GeoDataFrame, segment_length:int, Method = 'Vector', maxdt = 48, ui= False, vi= False)-> tuple[pd.DataFrame, list[int]]:
    """Calculates Autocorrilation on the entire dFAD dataset, producing a pandas dataframe of Tau, and Corrilations """
    segment_length = segment_length #length of the segment
    Tau_list = []
    R_list = []
    Ru_list = []
    Rv_list = []
    Rspeed_list = []
    traj_lengths = []
    traj_idx = []
    idx = 0
    for dFAD in range(len(dFADs)):
        trajectories = split_trajectory(dFADs.at[dFAD, 'x_speed'], dFADs.at[dFAD, 'y_speed'], dFADs.at[dFAD, 'TimeStamp'])
        if ui ==True:
            trajectories.v = trajectories.v*0
            trajectories.U.imag = trajectories.U.imag*0
        if vi ==True:
            trajectories.u = trajectories.u*0
            trajectories.U.real = trajectories.U.real*0
        trajectories.splice_dt_toolarge(maxdt= maxdt)
        for i in range(len(trajectories.U)): ## amount of split trajecotries
            if len(trajectories.U[i]) <= 1: ## make sure there are at least 2 points 
                continue
            ## evenly breakup trajectories 
            ac = Autocorrilation(trajectories.u[i],trajectories.v[i],trajectories.t[i])
            ac.Normalize_t(dt = 4)
            ac.splice_even_trajectories(segment_length)
            if ac.tooshort == True: ## skips this dFAD if entire Traj is shorten than the segment length
                continue
            for n in range(len(ac.U)):  
                acn = Autocorrilation(ac.u[n], ac.v[n], ac.t[n])  
                acn.dt = ac.dt  # carry over dt 
                acn.trajectory_length()
                acn.calc_autocor_allTau(method= Method) #Method : "Series" uses pandass autocorr function, "Vector" uses Autocorr function from Paquiero et al 2007
                traj_idx.extend([idx]*len(acn.Tau))
                idx += 1
                Tau_list.extend(acn.Tau)
                traj_lengths.append(acn.traj_length)
                if Method == 'Vector':
                    R_list.extend(acn.R)

                if Method == 'Series':
                    Ru_list.extend(acn.Ru)
                    Rv_list.extend(acn.Rv)
                    Rspeed_list.extend(acn.Rspeed)
                
    if Method == 'Vector':
        return pd.DataFrame({"Tau": Tau_list, "R": R_list}, index = traj_idx), traj_lengths
    if Method == 'Series':
        return pd.DataFrame({'Tau': Tau_list, 'Ru': Ru_list, 'Rv':Rv_list, 'Rspeed': Rspeed_list}, index = traj_idx), traj_lengths