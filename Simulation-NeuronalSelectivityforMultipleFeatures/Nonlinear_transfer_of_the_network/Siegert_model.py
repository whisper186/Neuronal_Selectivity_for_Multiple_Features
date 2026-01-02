####### Wenqing Wei ###############################################
####### The Siegert fomular (analytical solution) #################
####### to calculate the mean firing rate of a V1 #################
####### neuron depending on its input. ############################
###################################################################

import time
import numpy as np
from scipy.special import erfcx
import scipy.sparse

from importlib import reload
import output; reload(output); import output as out
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parameters; reload(parameters); import parameters as par

class Siegert_model_class():

    def loadConn(self):
        '''load the connectivity file of V1 neurons and fill them into matrix of N * N with 0 (not connected) and 1s (connected).'''
        self.connectivity = scipy.sparse.lil_matrix((self.N, self.N))
        
        idxs_e = np.load(self.sub_folder + 'conn_e.npy')
        idxs_i = np.load(self.sub_folder + 'conn_i.npy')
        
        for i in range(self.N):
            self.connectivity[i, idxs_e[i, :]] = self.PSP_e
            self.connectivity[i, idxs_i[i, :]] = self.PSP_i
        
        self.connectivity = self.connectivity.tocsr()
        self.connectivity2 = self.connectivity.multiply(self.connectivity)
    
    def loadConnFFIV1(self):
        '''load the connectivity file from FFI to V1 neurons and fill them into the matrix of N * N_FFI with 0 and 1s.'''
        self.FFIconnectivity = np.zeros((self.N, self.N_FFI))
        
        idxs = np.load(self.sub_folder + 'conn_FV.npy')
        
        for i in range(self.N):
            self.FFIconnectivity[i, idxs[i, :]] = self.PSP_FFI
        
        self.FFIconnectivity2 = np.multiply(self.FFIconnectivity, self.FFIconnectivity)
        
    
    def FFI_rate(self, Nu_FFI):
        '''Nu_FFI is the rate of compound thalamic input to single FFI neurons.'''
        mu_FFI = self.PSP_LGN_FFI * self.tau_m * Nu_FFI
        sig2_FFI = self.PSP_LGN_FFI**2 * self.tau_m * Nu_FFI
        sig_FFI = np.sqrt(sig2_FFI)
        
        low_lim_FFI = (self.V_r - mu_FFI)/sig_FFI
        up_lim_FFI = (self.V_th - mu_FFI)/sig_FFI
        
        integral_FFI = np.empty(mu_FFI.size)  
        
        for i in range(self.N_FFI):
            integral_FFI[i] = scipy.integrate.quad(lambda x: erfcx(-x), low_lim_FFI[i], up_lim_FFI[i])[0]
            if integral_FFI[i] == np.nan or integral_FFI[i] is None:
                raise UserWarning('there you go')
        
        Phi_FFI = self.tau_ref + self.tau_m * np.sqrt(np.pi) * integral_FFI
        
        return 1./Phi_FFI
    
    def Phi(self, v, Nu, V_FFI):
        '''
        Four parts: recurrent input + external background + exc. LGN input + inh. FFI input
        V_FFI: the rate of compound inhibitory FFI signal
        '''
        tmp_mu = self.connectivity.dot(v) + self.PSP_ext * self.bg_rate + self.PSP_LGN * Nu + self.FFIconnectivity.dot(V_FFI)
        mu = self.tau_m * tmp_mu
        
        tmp_sig2 = self.connectivity2.dot(v) + self.PSP_ext**2 * self.bg_rate + self.FFIconnectivity2.dot(V_FFI) + self.PSP_LGN**2 * Nu
        
        sig2 = self.tau_m * tmp_sig2
        
        sig = np.sqrt(sig2)
        
        low_lim = (self.V_r - mu)/sig
        up_lim = (self.V_th - mu)/sig
        
        integral = np.empty(mu.size)
        
        for i in range(self.N):
            integral[i] = scipy.integrate.quad(lambda x: erfcx(-x), low_lim[i], up_lim[i])[0]
            if integral[i] == np.nan or integral[i] is None:
                raise UserWarning('there you go')
        return self.tau_ref + self.tau_m * np.sqrt(np.pi) * integral
    
    def F(self, v, Nu, V_FFI):
        Phi = self.Phi(v, Nu, V_FFI)
        
        return 1./Phi
    
