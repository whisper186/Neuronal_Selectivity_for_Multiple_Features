######## Wenqing Wei ####################
######## functions for safety margin network simulation ##############

import numpy as np
import random
import matplotlib.pyplot as plt; plt.ion()
import time



def get_random_points(n, rf, mu):
	nps = np.empty((n,2))
	for i in range(n):
		r = rf/2. * np.sqrt(random.random())#np.sqrt(random.random())
		alpha = 2 * np.pi * random.random()
		x = r * np.cos(alpha) + mu[0]
		y = r * np.sin(alpha) + mu[1]
		nps[i,0] = x
		nps[i,1] = y
	return nps


def nearest_idx(nps, single_V1, nn):
    ''' get the nn nearest indices of LGN neurons to a single V1 neuron.'''
    #distance = np.linalg.norm(nps - single_V1.reshape((1,2)), axis = 1)
    c = nps - single_V1.reshape((1,2))
    distance = np.sqrt(np.einsum('ij, ij->i', c, c))
    nearest_id = distance.argsort()[:nn]
    return nearest_id


def get_connectivity_matrix(nps, V1_locs, near=True, balanced=True):
    connmatrix = np.zeros((len(V1_locs), len(nps)))
    if near==True and balanced==True:
        for i in range(12500):
            onid = nearest_idx(nps[:int(len(nps)/2)], V1_locs[i], 50)
            offid = nearest_idx(nps[int(len(nps)/2):], V1_locs[i], 50) + int(len(nps)/2)
            connmatrix[i, onid] = 1
            connmatrix[i, offid] = 1
    if near==True and balanced==False:
        for i in range(12500):
            idx = nearest_idx(nps, V1_locs[i], 100)
            connmatrix[i, idx] = 1
    return connmatrix



def idx_inside_rf(nps, single_V1, rf):
    ''' get the ids of the LGN neurons that is inside the receptive field of a V1 neuron.'''
    c = nps - single_V1.reshape((1, 2))
    dist = np.sqrt(np.einsum('ij, ij->i', c, c))
    idx = np.where(dist <= rf/2.)[0]
    return idx

def get_conn_matrix_in_rf(nps, V1_locs, rf):
    connmatrix = np.zeros((12500, len(nps)))
    for i in range(12500):
        idx = idx_inside_rf(nps, V1_locs[i], rf)
        connmatrix[i, idx] = 1
    return connmatrix

'''
nps = get_random_points(3071, 133., (0.,0.))
V1_locs = get_random_points(12500, 133.-17*2, (0.,0.))
connLV = get_connectivity_matrix(nps, V1_locs, near=True, balanced=False)
'''




#########################################################################
# calculate input compound input  #######################################
#########################################################################

def mcontrast(Fre_spa, sigv, c):
    sf = 2 * np.pi * Fre_spa
    return np.exp(-sf ** 2 * sigv[0]**2/2.) - c * np.exp(-sf**2 * sigv[1] **2/2.)


def k_vector(Fre_spa, theta):
    ''' calculate k vector which includes information of spatial frequency and stimulus orientation. '''
    sf = 2 * np.pi * Fre_spa
    k = sf * np.array([np.sin(theta), - np.cos(theta)]).reshape((2,1))
    return k

def get_alphas(nps, theta, Fre_spa):
    ''' calculate alphas and cosine of differences of alphas. '''
    sf = 2 * np.pi * Fre_spa
    k = k_vector(Fre_spa, theta)
    alpha = np.dot(nps, k)
    diff_alpha = []
    for i in range(len(alpha) - 1):
        diff_alpha.append(alpha[i] - alpha[i + 1::])
    diff = np.concatenate(diff_alpha, axis = 0)
    cos_diff = np.cos(diff)
    return alpha, cos_diff


def compound_amplitude_phase(nps, A, c, Fre_spa, sigc, sigs, theta):
    ''' calculate the amplitude and phase of the compound signal, which is the linear sum of the group of LGN neurons. '''
    m = mcontrast(Fre_spa, (sigc, sigs), c)
    if len(nps)==1:
        amplitude = m*A
        k = k_vector(Fre_spa, theta)
        phase = np.dot(nps, k)
    
    if len(nps)==0:
        amplitude, phase = 0,0


    if len(nps)>=2:
        alpha, cos_diff = get_alphas(nps, theta, Fre_spa)
        amplitude = m * A * np.sqrt(len(nps) + 2 * np.sum(cos_diff))
        x = np.sum(np.cos(alpha))
        y = np.sum(np.sin(alpha))
        if x < 0:
            phase = np.arctan(y/x) + np.pi
        else:
            phase = np.arctan(y/x)

    return amplitude, phase

def sep_compound_amplitude_phase(nps_ON, nps_OFF, A, c, Fre_spa, sigc, sigs, theta):
    ''' calculate the amplitude and phase of the compound signal including the excitatory and inhibitory groups. '''

    amp_ON, phase_ON = compound_amplitude_phase(nps_ON, A, c, Fre_spa, sigc, sigs, theta)
    amp_OFF, phase_OFF = compound_amplitude_phase(nps_OFF, A, c, Fre_spa, sigs, sigc, theta)
        
    sep_amp = np.sqrt(amp_ON ** 2 + amp_OFF ** 2 + 2 * amp_ON * amp_OFF * np.cos(phase_ON - phase_OFF))
    
    x = amp_ON * np.cos(phase_ON) + amp_OFF * np.cos(phase_OFF)
    y = amp_ON * np.sin(phase_ON) + amp_OFF * np.sin(phase_OFF)
    
    if x < 0:
        sep_phase = np.arctan(y/x) + np.pi
    else:
        sep_phase = np.arctan(y/x)

    return sep_amp, sep_phase


def input_thalamic_compound_input(nps, V1_locs, A=1, Fre_spa=0.08, sigc=1, sigs=5, n_lv=100):
    '''Calculate the amplitude and phase of the compound signal for all V1 neurons across all simulation degrees.'''
    t0 = time.time()
    amps = np.empty((len(V1_locs), 12))
    phases = np.empty((len(V1_locs), 12))
    deg_range = np.arange(0., 180., 15)
    for i in range(len(V1_locs)):
        #idx = np.where(connmatrix[i] ==1)[0]
        idx = nearest_idx(nps, V1_locs[i], int(n_lv))
        eid = idx[idx<int(len(nps)/2)]
        iid = idx[idx>=int(len(nps)/2)]
        
        for j in range(len(degree)):
            theta = deg_range[j]*np.pi/180
            sep_amp, sep_phase = sep_compound_amplitude_phase(nps[eid], nps[iid], A, 1., Fre_spa, sigc, sigs, theta)
            amps[i,j] = sep_amp
            phases[i,j] = sep_phase
        if (i+1)%1250 ==0:
            t1 = time.time()
            print(t1-t0)
    return amps, phases
    

def input_FFI_amp_phase(pv_locs, nps, A=1, c=1, Fre_spa=0.08, sigc=1, sigs=5, n_lp=8):
    '''Calculate the amplitude and phase of the compound signal for all FFI neurons across all simulation degrees.'''
    amps = np.empty((len(pv_locs), 12))
    phases = np.empty((len(pv_locs), 12))
    t0 = time.time()
    degree = np.arange(0., 180., 15)
    for i in range(len(pv_locs)):
        ids = nearest_idx(nps, pv_locs[i], int(n_lp))
        on_ids = ids[ids<int(len(nps)/2)]
        off_ids = ids[ids>=int(len(nps)/2)]
        for j in range(12):
            theta = degree[j] * np.pi / 180.

            sep_amp, sep_phase = sep_compound_amplitude_phase(nps[on_ids], nps[off_ids], A, c, Fre_spa, sigc, sigs, theta)
            
            amps[i,j] = sep_amp
            phases[i,j] = sep_phase
        
        if (i + 1)%1250 == 0:
            t1 = time.time()
            print (t1 - t0)
    return amps, phases


def V1_OSI_PO(V1_rates):
    OSIs = []
    POs = []
    deg_range = np.arange(0.,180.,15)
    for i in range(len(V1_rates)):
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


'''
locs = np.load('data/SafetyMargin_17_V1_FFI/input_locs.npy', allow_pickle=True)
V1_locs = locs[0]['V1_locs']
LGN_locs = locs[0]['LGN_locs']
FFI_locs = locs[0]['FFI_locs']

#amps, phases = input_thalamic_compound_input(LGN_locs, V1_locs, A=1, Fre_spa=0.08, sigc=1, sigs=5)
#FFI_amps, FFI_phases = input_PV_amp_phase(FFI_locs, LGN_locs, A=1, c=1, Fre_spa=0.08, sigc=1, sigs=5, n_lp=8)
'''

'''
# Generate 10 seeds for LGN positions
for i in range(10):
    nps = get_random_points(3071, 133., (0.,0.))
    np.save('data/SafetyMargin_17_V1_FFI/LGN_pos_seed/LGN_locs_%i.npy'%i, nps)
    print (np.max(nps), np.min(nps))
    amps, phases = input_thalamic_compound_input(nps, V1_locs, A=1, Fre_spa=0.08, sigc=1, sigs=5)
    np.save('data/SafetyMargin_17_V1_FFI/LGN_pos_seed/amps_1-5_0.08_1_LGNseed_%i.npy'%i, amps)
    np.save('data/SafetyMargin_17_V1_FFI/LGN_pos_seed/phases_1-5_0.08_1_LGNseed_%i.npy'%i, phases)

'''


'''
# Generate the amplitudes and phases of compound inputs to F1 and V1 neurons for different spatial frequencies.
sFs = np.array([0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4])

for i in range(len(sFs)):
    amps, phases = input_thalamic_compound_input(LGN_locs, V1_locs, A=1, Fre_spa = sFs[i], sigc = 1., sigs = 5., n_lv=100)
    FFI_amps, FFI_phases = input_FFI_amp_phase(FFI_locs, LGN_locs, A=1, Fre_spa = sFs[i], sigc = 1., sigs = 5., n_lp = 8)
    np.save('data/SafetyMargin_simu_sFs/sFs_amps_phases/amps_%.3f_1-5_1.npy'%(sFs[i]), amps)
    np.save('data/SafetyMargin_simu_sFs/sFs_amps_phases/phases_%.3f_1-5_1.npy'%(sFs[i]), phases)
    np.save('data/SafetyMargin_simu_sFs/sFs_amps_phases/FFI_amps_%.3f_1-5_1.npy'%(sFs[i]), FFI_amps)
    np.save('data/SafetyMargin_simu_sFs/sFs_amps_phases/FFI_phases_%.3f_1-5_1.npy'%(sFs[i]), FFI_phases)
'''

'''
# Generate the amplitudes and phases of compound inputs to V1 neurons for different convergent number from LGN --> V1.
n_lvs = np.array([2, 10, 20, 50, 100, 200, 300, 400, 500, 1000])
As = 10000./n_lvs
for i in range(len(n_lvs)):
    LGN_locs = np.load('data/SafetyMargin_simu_convergence/LGN_locs/LGN_locs_sumN_%i.npy'%n_lvs[i])
    amps, phases = input_thalamic_compound_input(LGN_locs, V1_locs, A=1, Fre_spa = 0.08, sigc=1., sigs=5., n_lv = n_lvs[i])
    np.save('data/SafetyMargin_simu_convergence/convergence_amps_phases/amps_sumN_%i_1-5_1.npy'%(n_lvs[i]), amps)
    np.save('data/SafetyMargin_simu_convergence/convergence_amps_phases/phases_sumN_%i_1-5_1.npy'%(n_lvs[i]), phases)

'''


