####### Wenqing Wei ###############################################
####### Import parameters and combine the dynamic rate model ######
####### and predictor classes. ####################################
####### Prepare for fullSolution. #################################
###################################################################

import numpy as np
from importlib import reload
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parameters; reload(parameters); import parameters as par

import scipy.sparse
from multiprocessing import Queue, Process

import Siegert_model; reload(Siegert_model); from Siegert_model import Siegert_model_class
import Siegert_predictors; reload(Siegert_predictors); from Siegert_predictors import Siegert_predictor_class

#import matplotlib.pyplot as plt

class single_neuron_mf(Siegert_predictor_class, Siegert_model_class):

    def __init__(self):
        '''
        Mind: everything here is working in === s/Hz ===
        '''

        self.populations = ['exc', 'inh']

        ### neuron model parameters
        self.tau_ref =  par.t_ref * 1e-3 # s 2.0
        self.tau_m =  par.tau_m * 1e-3   # s 20.
        
        self.V_th = par.V_th             # mV 20.
        self.V_r = par.V_reset           # mV 0.

        self.C_m = par.C_m                  # pF
        self.R = self.tau_m/self.C_m * 1e3    # GOhm

        ### connectivity parametes
        self.n_neurons = np.array([par.Ne, par.Ni])  # Nr. of exc. and inh. Neurons (10000, 2500)
        self.N = self.n_neurons.sum()   # Nr. of total neuron in the network (12500)
        
        self.N_FFI = int(par.N_FFI)    # Nr. of FFI neurons in the network 1128

        self.K_e = int(par.ind_E)   # recurrent exc. inputs to individual V1 1000
        self.K_i = int(par.ind_I)   # recurrent inh. inputs to individual V1 250
        
        self.K_FFI = int(par.n_FFI_V1)  

        self.g = par.g              # inhibition-excitation ratio  -8
        self.PSP_e = par.J_E        # exc. synaptic weight 0.2
        self.PSP_i = self.g* self.PSP_e    # inh. synaptic weight 0.2 * -8
        
        self.PSP_FFI = par.J_FFI_V1       # synaptic weight from FFI --> V1 
        self.PSP_LGN = par.J_LV#f * par.g_E   # thalamocortical synaptic weight
        self.PSP_LGN_FFI = par.J_LF#g_FFI       # synaptic weight frm LGN --> FFI

        self.PSP_ext = par.J_ext           # external input weight
        self.bg_rate = par.ext_rate        # external input rate, Hz

        ### connectivity
        self.connectivity = None
        self.connectivity2 = None

        ### input
        self.n_lf = par.n_LF
        self.n_lv = par.n_LV
        
        self.n = par.n           # Nr. of thalamic inputs to individual V1
        self.A = par.A           # Mean firing rate of LGN neuron
        self.m = par.mmax

        self.sf = par.Fre_spa    # spatial frequency of grating
        self.tf = par.Fre_tem    # temporal frequency of grating
        self.RF = np.array([1., 5.])  # sigmas of receptive field of LGN neuron

        self.sub_folder = '../input_data/'

