######## Wenqing Wei ####################
# network configuration for instantaneous rate of dLGN cells
# NEST 2.20
######## Network with safety margin network simulation ##############
######## run the network with instantaneous rate of LGN cells to draw the receptive fields ##########



import numpy as np
import random
import nest
import time

from importlib import reload
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parameters; reload(parameters); import parameters as par

import Response_to_visual_stimuli; reload(Response_to_visual_stimuli);
from Response_to_visual_stimuli import load_locs_and_connections


class inhomo_neural_Network(load_locs_and_connections, ):

    def __init__(self):
        
        # load locations of neurons
        self.locs = self.load_locations()
        self.V1_locs = self.locs['V1_locs']
        self.dLGN_locs = self.locs['dLGN_locs']
        self.FFI_locs = self.locs['FFI_locs']

        # load connections between neuron groups
        self.connections = self.load_connections('all')
        self.conn_e = self.connections['rec_e']
        self.conn_i = self.connections['rec_i']
        self.conn_LV = self.connections['dLGNtoV1']
        self.conn_FV = self.connections['FFItoV1']
        # parameters for dLGN neurons
        self.n = par.n
        self.all_n = len(self.dLGN_locs)
        
        # parameters for network simulation
        self.tau_m = par.tau_m
        self.t_ref = par.t_ref
        self.delay = par.delay
        self.V_rest = par.E_L
        self.V_reset = par.V_reset
        self.V_th = par.V_th
        self.dt = par.dt
        self.A = par.A

        self.N_FFI = len(self.FFI_locs)
        self.N = len(self.V1_locs)
        self.fe = par.fe
        self.Ne = int(self.N*self.fe)
        self.Ni = int(self.N-self.Ne)
        self.ext_rate = par.ext_rate
        self.J_ext = par.J_ext  # external synaptic weight
        self.J_E = par.J_E    # recurrent excitatory connection weight
        self.g = par.g       # excitation-inhibition ratio
        self.J_I = self.J_E * self.g

        self.p = par.p      # recurrent connectivity
        self.ind_E = int(self.p*self.Ne)
        self.ind_I = int(self.p*self.Ni)

        self.n_LV = par.n_LV
        self.n_LF = par.n_LF
        self.n_FV = par.n_FFI_V1
        self.J_LV = par.J_LV
        self.J_LF = par.J_LF
        self.J_FV = par.J_FFI_V1
        
        self.t_drop = 200.   # discard the first 200 ms
        self.sf = par.Fre_spa
        self.tf = par.Fre_tem
        self.n_vp = par.n_vp   # number of virtual processes
        self.mmax = par.mmax
        self.time_bin = 200.


    def nearest_idx(self, nps, single_V1, nn):
        ''' get the nn nearest indices of LGN neurons to a single V1 neuron.'''
        c = nps - single_V1.reshape((1, 2))
        distance = np.sqrt(np.einsum('ij, ij->i', c, c))
        nearest_id = distance.argsort()[:nn]
        return nearest_id
    
    def inhomo_network_simulation(self, rate_times, rate_values, ext_rate):
        nest.ResetKernel()
        nest.SetKernelStatus({'overwrite_files': True,
                            'total_num_virtual_procs': self.n_vp,
                            'resolution': self.dt})
        nest.SetKernelStatus({'print_time': True})
        
        # set seed =========================================================
        a = 2
        msd = 1234 * a * 100 + 1
        n_vp_g = nest.GetKernelStatus('total_num_virtual_procs')
        msdrange2 = range(msd + n_vp_g + 1, msd + 2 * n_vp_g + 1)
        nest.SetKernelStatus({'rng_seeds' : msdrange2,
                            'grng_seed' : msd + n_vp_g})
        # ==================================================================
        
        # Create V1 nodes
        V1_nodes = nest.Create('iaf_psc_delta', self.N, params = {'tau_m': self.tau_m,
                                                                  't_ref': self.t_ref,
                                                                  'V_th': self.V_th,
                                                                  'V_reset': self.V_reset,
                                                                  'E_L': self.V_rest})
        V1_E, V1_I = V1_nodes[:self.Ne], V1_nodes[self.Ne:]

        # Create external input and connect to V1 neurons
        external_input = nest.Create('poisson_generator', 1, {'rate': ext_rate})
        syn_ext = {'model': 'static_synapse', 'delay': self.delay, 'weight': self.J_ext}
        nest.Connect(external_input, V1_nodes, syn_spec = syn_ext)

        ####################################################################
        ### recurrent connections ##########################################
        ####################################################################
        list_exc_pre = list(np.asarray(V1_nodes)[conn_e.ravel()])
        list_exc_post = list(np.repeat([V1_nodes], int(self.ind_E)))
        syn_dict_exc = {'model': 'static_synapse', 'delay': self.delay, 'weight': self.J_E}
        nest.Connect(list_exc_pre, list_exc_post, 'one_to_one', syn_spec = syn_dict_exc)

        list_inh_pre = list(np.asarray(V1_nodes)[conn_i.ravel()])
        list_inh_post = list(np.repeat([V1_nodes], int(self.ind_I)))
        syn_dict_inh = {'model': 'static_synapse', 'delay': self.delay, 'weight': self.J_I}
        nest.Connect(list_inh_pre, list_inh_post, 'one_to_one', syn_spec = syn_dict_inh)

        print ('Connect recurrent and external input -- Finish')

        ####################################################################
        ### Create LGN sensors #############################################
        ####################################################################
        lgn_sensors = nest.Create('inhomogeneous_poisson_generator', self.all_n)
        for i in range(self.all_n):
            nest.SetStatus(lgn_sensors[i:i+1],
                           {'rate_times': rate_times.tolist(),
                            'rate_values': rate_values[:, i].tolist()})
        print ('set status to dLGN sensors \n mean: %.3f'%(np.mean(rate_values)))

        parrot_lgn = nest.Create('parrot_neuron', len(self.dLGN_locs))
        syn_dict_parrot = {'model': 'static_synapse', 'weight': 1.0, 'delay': self.delay}
        nest.Connect(lgn_sensors, parrot_lgn, 'one_to_one', syn_spec = syn_dict_parrot)

        print ('Create parrot and connect to dLGN sensors -- Finish')

        # dLGN --> V1 nodes
        lgn_V1_list, V1_list = [], []
        for ii in range(self.N):
            idx = self.nearest_idx(self.dLGN_locs, self.V1_locs[ii], int(self.n_LV))
            lgn_V1_idx = idx
            lgn_V1_list += list(np.asarray(parrot_lgn)[lgn_V1_idx])
            V1_list += [V1_nodes[ii]] * int(self.n_LV)
        syn_lgn_V1 = {'model': 'static_synapse', 'delay': self.delay, 'weight': self.J_LV}
        nest.Connect(lgn_V1_list, V1_list, 'one_to_one', syn_spec = syn_lgn_V1)

        print ('Connect parrot dLGN to V1 -- Finish')

        ####################################################################
        ### Create FFI neurons and connect LGN neuron to them ##############
        ####################################################################
        FFI_nodes = nest.Create('iaf_psc_delta', int(self.N_FFI),
                                params = {'tau_m': self.tau_m,
                                          't_ref': self.t_ref,
                                          'V_th': self.V_th,
                                          'V_reset': self.V_reset,
                                          'E_L': self.V_rest,})
        # parrot dLGN --> FFI
        lgn_ffi_list, ffi_list = [], []
        for i in range(int(self.N_FFI)):
            ids = self.nearest_idx(self.dLGN_locs, self.FFI_locs[i], int(self.n_LF))
            lgn_ffi_list += list(np.asarray(parrot_lgn)[ids])
            ffi_list += [FFI_nodes[i]] * int(self.n_LF)
        syn_lgn_ffi = {'model': 'static_synapse', 'delay': self.delay, 'weight': self.J_LF}
        nest.Connect(lgn_ffi_list, ffi_list, 'one_to_one', syn_spec = syn_lgn_ffi)

        print ('Connect parrot dLGN to FFI -- Finish')

        ####################################################################
        ### Connect FFI neurons to V1 neurons ##############################
        ####################################################################
        list_FFI = list(np.asarray(FFI_nodes)[self.conn_FV.ravel()])
        list_V1 = list(np.repeat([V1_nodes], self.n_FV))
        syn_FFI_V1 = {'model': 'static_synapse', 'delay': self.delay, 'weight': self.J_FV}
        nest.Connect(list_FFI, list_V1, 'one_to_one', syn_spec = syn_FFI_V1)

        print ('Connect FFI to V1 -- Finish')

        #####################################################################
        spikedetector = nest.Create('spike_detector', 1, {'start': self.t_drop})
        nest.Connect(V1_nodes, spikedetector)

        sd_dLGN = nest.Create('spike_detector', 1, {'start': self.t_drop})
        nest.Connect(parrot_lgn, sd_dLGN)

        sd_FFI = nest.Create('spike_detector', 1, {'start': self.t_drop})
        nest.Connect(FFI_nodes, sd_FFI)

        time_bin = rate_times[-1] - rate_times[-2]
        T = rate_times.max() + time_bin
        nest.Simulate(T)
        dSD = nest.GetStatus(spikedetector, keys = 'events')[0]
        dSD_dLGN = nest.GetStatus(sd_dLGN, keys = 'events')[0]
        dSD_FFI = nest.GetStatus(sd_FFI, keys = 'events')[0]
        print (len(dSD['times'])/T*1000./self.N)

        return dSD, dSD_dLGN, dSD_FFI





