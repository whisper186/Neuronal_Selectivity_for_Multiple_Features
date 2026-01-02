##### Wenqing Wei #############
##### get input current from spikes, seed 2 of network simulation with SafetyMargin

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import random
import time
import sys

def instantaneous_rate_of_inputs(ts, binsize):
    ''' Get instantaneous rate according to the spike train.'''
    spike_num, hbins = np.histogram(ts, binsize)
    step = binsize[1] - binsize[0]
    rate = spike_num * 1000./step
    return rate

def fourier_transform_to_temporal_signal(data):
    ''' Do the fourier transform to the temporal signal.'''
    t = np.linspace(0.2, 6.2, 301)[:-1]
    T = t[1] - t[0]
    N = data.size
    freq = np.linspace(0., 1/T, N)
    signal_fft = np.fft.fft(data)
    amplitudes = np.abs(signal_fft[:int(N/2)]) *2./N
    idx = (np.abs(freq-3.)).argmin()
    F1 = amplitudes[idx]
    phases = np.angle(signal_fft[:int(N/2)])
    phase = phases[idx]
    return F1, phase


def single_OSI(tc):
    '''Calculate the OSI and PO from the tuning curve of a single neuron.
    Parameters:
    tc: array, shape (12, ), the response rate of a single V1 neuron across all 12 stimulus orientations.'''
    deg_range = np.arange(0., 180., 15)
    OSV = np.sum(tc*np.exp(2*np.pi * 1j * deg_range/180.))
    nOSV = np.sum(tc)
    
    if nOSV == 0:
        OSV, nOSV = 0, 1
    OSVs = OSV/nOSV
    PO = np.angle(OSVs)
    if PO < 0:
        PO = PO + 2*np.pi
    PO = PO/(2*np.pi) * 180.
    OSI = abs(OSVs)
    return OSI, PO


def V1_OSI_PO(V1_rates):
    '''Calculate the OSI and PO from the tuning curve for all V1 neurons.
    Parameters:
    V1_rates: array, shape (n, 12), the response rates of V1 neurons across all 12 stimulus orientations.'''
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


conn_LV = np.load('input_data/conn_LV.npy')
conn_FFI = np.load('input_data/conn_FFI.npy')

angles = np.arange(0, 180, 15)
binsize = np.linspace(200., 6200., 301)
Cm = 250  # pF
deg_range = np.arange(0., 180., 15)
ids = [39, 3013, 2471, 11335, 12370]
sd = int(2)  # seed number

path = 'output_data/simu_default_params/seed_%i/'%(sd)
folder = path + 'seed_%i_current_input/'%(sd)

if not os.path.exists(folder):
        os.makedirs(folder)

if not os.path.exists(path + 'V1_rates.npy'):
    import parameters; reload(parameters); from parameters import *
    import network_configuration; reload(network_configuration);
    import network_configuration as network

    V1_rates = network.get_V1_neuron_rates(path, network.V1_locs, network.dLGN_locs, network.FFI_locs, ext_rate, a=sd)
    print ('seed %i simulation - Finished!'%(sd))

def presynaptic_input_current(source, inpts, inpevs, J):
    pre_spk = [inpts[inpevs == x] for x in source]
    pre_spk = np.concatenate(pre_spk, axis = 0)
    pre_rate = instantaneous_rate_of_inputs(pre_spk, binsize)
    current = J * 1e-3 * Cm * pre_rate
    return current

def calculate_input_current(deg, V1_id, J_exc = 2., J_inh = -1.6):
    input_ts_evs = np.load(path + 'input_ts_evs_deg_%i.npy'%deg)
    ints, inevs = input_ts_evs[:,0], input_ts_evs[:,1]
    
    esource = conn_LV[V1_id]
    ecurrent = presynaptic_input_current(esource+15573, ints, inevs, J_exc)
    
    isource = conn_FFI[V1_id]
    icurrent = presynaptic_input_current(isource+18644, ints, inevs, J_inh)
    return ecurrent + icurrent

def get_input_mean_F1(V1_id):
    means, F1s = [], []
    for i in deg_range:
        current = calculate_input_current(i, V1_id)
        means.append(np.mean(current))
        
        F1, phase = fourier_transform_to_temporal_signal(current)
        F1s.append(F1)
    return means, F1s


def generate_data_for_figure(ids=ids):
    
    io_id_mean_F1_rates = np.empty((4, len(deg_range), len(ids)))
    io_id_mean_F1_rates[0,:,:] = ids
    V1_rates = np.load(path + 'V1_rates.npy')

    for i in range(len(ids)):
        means, F1s = get_input_mean_F1(ids[i])
        io_id_mean_F1_rates[1,:,i] = means
        io_id_mean_F1_rates[2,:,i] = F1s
        io_id_mean_F1_rates[3,:,i] = V1_rates[ids[i]]
    np.save(folder + 'io_id_mean_F1_rates.npy', io_id_mean_F1_rates)



