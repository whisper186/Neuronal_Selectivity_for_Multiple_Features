####### Wenqing Wei ###############################################
####### Analyse the output data and save the data #################
####### that used to plot the figures in paper. ###################
###################################################################

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import sys, time
from importlib import reload
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parameters; reload(parameters); import parameters as par

angles = np.arange(0., 180., 15)

save_path = 'output_data/'
load_path = '../input_data/'

conn_FFI = np.load(load_path + 'conn_FV.npy')
conn_e = np.load(load_path + 'conn_e.npy')
conn_i = np.load(load_path + 'conn_i.npy')

if not os.path.exists(save_path):
    print ('No output data! You should first run the analytical simulation!')
    import fullSolution; reload(fullSolution)

v_all_full = np.load(save_path + 'full_rates_nt60.npy')
lgn_all_input = np.load(save_path + 'lgn_input_nt60.npy')
vFFI_all_full = np.load(save_path + 'FFI_rates_nt60.npy')

# calculate the feedforward inhibitory input current
fac_inh = par.C_m * par.J_FFI_V1 * 1e-3
I_inh = fac_inh * vFFI_all_full

def get_inh_input_current(idxs):
    i_all = np.zeros((12, 60))
    for i in idxs:
        i_inh = I_inh[:, i, :]
        i_all += i_inh
    return i_all

I_inh_all = np.zeros((12, par.N, 60))
t0 = time.time()
for ids in range(par.N):
    idxs = conn_FFI[ids]
    i_all = get_inh_input_current(idxs)
    I_inh_all[:, ids, :] = i_all
    if ids%1250 == 0:
        print (ids, time.time()-t0)

np.save(save_path + 'ffi_input_current.npy', I_inh_all)


# calculate the recurrent input
I_rec_all = np.empty((12, par.N, 60))
t0 = time.time()
for i in range(par.N):
    I_rec = np.empty((12, 1250, 60))
    for eid in range(1000):
        I_rec[:, eid, :] = par.C_m * par.J_E * 1e-3 * v_all_full[:, conn_e[i, eid], :]
    for iid in range(250):
        I_rec[:, 1000+iid, :] = par.C_m * par.J_I * 1e-3 * v_all_full[:, conn_i[i, iid], :]
    I_rec_all[:, i, :] = np.sum(I_rec, axis = 1)
    if (i+1)%1250==0:
        print(i, time.time()-t0)

np.save(save_path + 'recurrent_input_current.npy', I_rec_all)


# calculate the feedforward and all input current and save
I_ff_inh_input = np.load(save_path + 'ffi_input_current.npy')
I_rec_input = np.load(save_path + 'recurrent_input_current.npy')
I_lgn_input = par.C_m * par.J_LV * 1e-3 * lgn_all_input
I_ff_input = I_lgn_input + I_ff_inh_input
I_all_input = I_lgn_input + I_ff_inh_input + I_rec_input + par.ext_rate * par.J_ext * par.C_m*1e-3

np.save(save_path + 'ff_input_current.npy', I_ff_input)
np.save(save_path + 'all_input_current.npy', I_all_input)


def V1_OSI_PO(V1_rates):
    OSIs = []
    POs = []
    deg_range = np.arange(0.,180.,15)
    for i in range(12500):
        OSV = np.sum(V1_rates[i] * np.exp(2 * np.pi * 1j * deg_range/180.))
        nOSV = np.sum(V1_rates[i])
        
        if nOSV == 0:
            OSV, nOSV = 0,1
            
        OSVs = OSV/nOSV
        PO = np.angle(OSVs)
        if PO < 0:
            PO = PO + 2 * np.pi
        PO = PO/(2 * np.pi) * 180.
        OSI = abs(OSVs)
        POs.append(PO)
        OSIs.append(OSI)
    return OSIs, POs

transFunc_V1_rates = v_all_full.mean(axis = 2).T

transFunc_OSI, transFunc_PO = V1_OSI_PO(transFunc_V1_rates)
np.save(save_path + 'transFunc_OSI_PO.npy', np.array([transFunc_OSI, transFunc_PO]).T)

num_path = '../output_data/simu_spatialFrequencies/sFs_amps_phases/'
com_rates= np.load('com_spk_rates_%.3f.npy'%(0.08))
cOSIs, cPOs = V1_OSI_PO(com_rates)
np.save(save_path + 'num_com_V1_OSI_PO.npy', np.array([cOSIs, cPOs]).T)