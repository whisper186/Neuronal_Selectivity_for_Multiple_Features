####### Wenqing Wei ###############################################
####### Plot the compound thalamic inputs to a single V1 neuron. ##
####### Network Simulation ########################################
####### The simulation runs on NEST 2.20 ##########################
###################################################################


##################################################################
# run for different seeds ########################################
##################################################################
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import random
import os
from importlib import reload

from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.font_manager import FontProperties

import parameters; reload(parameters); from parameters import *
import network_configuration; reload(network_configuration);
import network_configuration as network

print ('run for different seeds')


folder = 'output_data/simu_default_params/'

if not os.path.exists(folder):
    os.makedirs(folder)

seedrange = np.arange(1, 51, 1)
for seed in seedrange:
    if not os.path.exists(folder + 'seed_%i'%seed):
        os.makedirs(folder + 'seed_%i'%seed)




seed_simulation = False
if seed_simulation == True:
    for seed in seedrange:
        savefolder = folder + 'seed_%i'%(seed)
        V1_rates = network.get_V1_neuron_rates(savefolder, network.V1_locs, network.dLGN_locs, network.FFI_locs, ext_rate, a=int(seed))
        print ('finished -- %i'%(seed))
    print ('All seed simulation -- Finished!')

deg_range = np.arange(0., 180., 15)

conn_LV = np.load('input_data/conn_LV.npy')

lgnidmin = N + 1 + all_n + 1
ffiidmin = lgnidmin + all_n
binsize = np.linspace(200., 6200., 301)
t = np.linspace(0.2, 6.2, 301)[:-1]
T = t[1] - t[0]
tN = len(t)
freq = np.linspace(0., 1/(2*T), int(tN/2))

def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def get_input_tc_spikes(V1_id, deg, folder):
    '''Extract the input spikes of dLGN cells to a given V1 cell.
    Return an array of spiking times of presynaptic dLGN cells. 
    Parameters:
    V1_id : int, the index of V1 cell. The minimum value is 0.
    deg : float or int, the orientation of stimulus grating.
    folder : the directory that saves the simulation data to be analysed.'''
    conn_lgn_Ctx = np.load('input_data/conn_LV.npy')
    spks = np.load(folder + '/input_ts_evs_deg_%i.npy'%deg)
    ts, evs = spks[:,0], spks[:,1]
    source = conn_lgn_Ctx[V1_id]
    lgnspks = np.array([ts[evs == x + lgnidmin] for x in source])
    return lgnspks  
    


def instantaneous_input_current_of_inputs(ts, T, J=2.):
    '''Calculate the instantaneous input current of the spike train.
    Return an array.
    Parameters:
    ts : an array, spike times.
    T : float or int, the binsize of PSTH, ms.
    J : float, the synaptic weight, mV.'''
    tN = 6000./T
    bs = np.linspace(200.05, 6200.+0.05, int(tN+1))
    spk_num, hbin = np.histogram(ts, bs)
    r = spk_num*1000./T
    inpt_current = r*1e-3*C_m*J
    return inpt_current


def fft_of_signal(ts, T):
    '''Compute the discrete fourier transform of the spike train. 
    Return the frequencies and amplitudes at the frequencies.
    Parameters:
    ts : array, the spike train.
    T : float or int, the binsize of PSTH, ms.'''
    tN = 6000./T
    inpt_current = instantaneous_input_current_of_inputs(ts, T)
    freq = np.linspace(0., 1/(2*T)*1000., int(tN/2))
    signal_fft = np.fft.fft(inpt_current)
    amplitudes = np.abs(signal_fft[:int(tN//2)]) * 2/tN
    return freq, amplitudes
    

def get_tuning_of_fft(V1_id, T, folder):
    '''Compute the discrete fourier transform of the spike train at all orientations.
    Return an array with shape (int(6000/(T*2)), 12) and frequencies.
    Parameters:
    V1_id : int, the index of V1 cell. The minimum value is 0.
    T : float or int, the binsize of PSTH, ms.
    folder : the directory that saves the simulation data to be analysed.
    '''
    tunings = np.empty((int(6000/(T*2)), 12))
    
    for i in range(12):
        inpt_spk = get_input_tc_spikes(V1_id, deg_range[i], folder)
        spks = np.concatenate(inpt_spk,  axis = 0)
        f, a = fft_of_signal(spks, T)
        tunings[:,i] = a
    return tunings, f


def get_tuning_of_fft_of_V1id(T, path, V1id = 2471):
    '''Compute the discrete fourier transform of the presynaptic spike trains that project to the V1 cell id V1id at all orientations running for 50 different simulation seeds.
    Return an array with shape (int(6000/(T*2)), 12, 50) and frequencies.
    Parameters:
    T : float or int, the binsize of PSTH, ms.
    '''
    tunings = np.empty((int(6000/(T*2)), 12, 50))
    for j in range(50):
        for i in range(12):
            inspks = get_input_tc_spikes(V1id, deg_range[i], path + 'seed_%i'%(j+1))
            ts = np.concatenate(inspks, axis = 0)
            f, a = fft_of_signal(ts, T)
            tunings[:, i, j] = a
        print ('seed -- %i'%(j+1))
    return tunings, f

def get_power_spectrum(all_seed_tunings):
    '''Extract the mean and standard deviation of all_seed_tunings over 50 simulation seeds at simulation orientation 60°.
    Return two arrays with shape (len(all_seed_tunings),).
    Parameters:
    all_seed_tunings : array with shape (int(6000/(T*2)), 12, 50).
    '''
    ave_seed_tunings = np.mean(all_seed_tunings, axis=2)
    std_seed_tunings = np.std(all_seed_tunings, axis=2)
    ave_seed_tunings[0,:] = ave_seed_tunings[0,:]/2
    ps = ave_seed_tunings[:,4]  # the average tuning curve at 60 degrees
    std = std_seed_tunings[:,4]
    
    return ps, std

def OSI_at_different_frequencies(tunings):
    '''Calculate the OSIs from the tuning curves at different frequencies.
    Return a list with length len(tunings).
    Parameters:
    tunings: array with shape (N, len(deg_range)).'''
    OSIs = []
    N = len(tunings)
    for i in range(N):
        if tunings[i].max() <= 0.:
            OSIs.append(0.)
        
        else:
            OSV = np.sum(tunings[i] * np.exp(2*np.pi*1j*deg_range/180.))
            nOSV = np.sum(tunings[i])
            
            if nOSV==0:
                OSV, nOSV = 0,1
            
            OSVs = OSV/nOSV
            OSI = abs(OSVs)
            OSIs.append(OSI)
    return OSIs



###################################################################
# plot functions ##################################################
###################################################################

def plot_raster_plot(gs, inpt_spk, ONidx, OFFidx):
    ax = plt.subplot(gs)

    for i in range(len(ONidx)):
        sid = np.where(idx==ONidx[i])[0][0]
        ax.plot(inpt_spk[sid], np.ones((len(inpt_spk[sid])))*(i), '|', markersize=3, color='brown')
    for j in range(len(OFFidx)):
        sid = np.where(idx==OFFidx[j])[0][0]
        ax.plot(inpt_spk[sid], np.ones((len(inpt_spk[sid])))*(j+len(ONidx)), '|', markersize=3, color='green')
    
    ax.set_xlim(200., 1200.)
    ax.set_ylim(0, 21)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel('Input LGN Neuron ID')
    hide_axis(ax)

def plot_instantaneous_current(gs, ts, T):
    ax = plt.subplot(gs)
    bs = np.linspace(200.05, 6200.05, int(6000./T+1))
    hs = ax.hist(ts, bs[:int(len(bs)/6)], facecolor='blue', edgecolor='none')
    y_vals = ax.get_yticks()
    ax.set_yticklabels(['%i'%(x/(2*T)) for x in y_vals])
    
    ax.set_xlim(200., 1200.)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(r'I$^\mathrm{LGN}$ [nA]')
    hide_axis(ax)

def plot_new_power_spectrum_ylog(fig, gs, all_seed_tunings):
    
    ave_seed_tuning = np.mean(all_seed_tunings, axis=2)
    std_seed_tuning = np.std(all_seed_tunings, axis=2)
    ave_seed_tuning[0,:] = ave_seed_tuning[0,:]/2
    ps = ave_seed_tuning[:,4]   # the average tuning curve at 60 degrees
    std = std_seed_tuning[:,4]
    
    
    gss = gs.subgridspec(1,2, width_ratios=[1, 20], wspace=0.08)
    
    ax00 = fig.add_subplot(gss[0,0])
    ax00.plot(freq[0], ps[0], '-o', markersize=3, clip_on=False)
    ax00.set_ylim(9., 4999)
    ax00.set_yscale('log')
    hide_axis(ax00)
    ax00.set_xlim(freq[0], 0.1)
    ax00.set_xticks([0.])
    
    ax01 = fig.add_subplot(gss[0,1])
    ax01.plot(freq[1:], ps[1:], lw=lw)
    ax01.fill_between(freq[1:], ps[1:]-std[1:], ps[1:]+std[1:], facecolor='lightgray', edgecolor='none', alpha=1)
    ax01.set_ylim(9., 4999)
    ax01.set_yscale('log')
    ax01.set_xscale('log')
    ax01.set_xlim(0.16, 5000)
    hide_axis(ax01)
    ax01.spines['left'].set_visible(False)
    ax01.tick_params(left=False, labelleft=False)
    ax01.yaxis.set_ticks_position('none')
    
    ax01.set_xticks([1., 3., 10, 100, 1000])
    ax01.set_xticklabels([r'$10^0$', 3, r'$10^1$', r'$10^2$', r'$10^3$'])
    d=0.01
    kwargs = dict(transform=ax00.transAxes, color='k', clip_on=False)
    ax00.plot((1-d*20, 1+d*20), (-d, +d), **kwargs)

    kwargs.update(transform=ax01.transAxes)
    ax01.plot((-d, +d), (-d, +d), **kwargs)

    ax01.set_xlabel('Frequency [Hz]')
    ax00.set_ylabel('Signal power [a.u.]')

def plot_new_fft_OSI(fig, gs):
    
    ave_seed_tuning = np.mean(all_seed_tunings, axis=2)
    ave_seed_tuning[0,:] = ave_seed_tuning[0,:]/2
    OSIs = OSI_at_different_frequencies(ave_seed_tuning)
    
    gss = gs.subgridspec(1,2, width_ratios=[1,20], wspace=0.08)

    ax00 = fig.add_subplot(gss[0,0])
    ax00.plot(freq[0],OSIs[0], '-o', markersize=3, clip_on=False)
    hide_axis(ax00)
    #ax00.set_xlim(-0.1, 0.1)
    ax00.set_xlim(freq[0], 0.1)
    ax00.set_ylim(-0.03, OSIs[18]*1.5)
    ax00.set_xticks([0.])

    ax01 = fig.add_subplot(gss[0,1])
    ax01.plot(freq[1:], OSIs[1:], lw=lw)
    ax01.set_xscale('log')
    ax01.set_xlim(0.16, 5000)
    ax01.set_ylim(-0.03, OSIs[18]*1.5)
    hide_axis(ax01)
    ax01.spines['left'].set_visible(False)
    ax01.tick_params(left=False, labelleft=False)
    ax01.set_xticks([1., 3., 10, 100, 1000])
    ax01.set_xticklabels([r'$10^0$', 3, r'$10^1$', r'$10^2$', r'$10^3$'])

    d=0.01
    kwargs = dict(transform=ax00.transAxes, color='k', clip_on=False)
    ax00.plot((1-d*20, 1+d*20), (-d, +d), **kwargs)

    kwargs.update(transform=ax01.transAxes)
    ax01.plot((-d, +d), (-d, +d), **kwargs)

    ax01.set_xlabel('Frequency [Hz]')
    ax00.set_ylabel('OSI')


dirpath = 'output_data/simu_default_params/'
V1id = 2471
filename = dirpath + 'fq_amps_all_seed_tunings_%i.npy'%V1id
if not os.path.exists(filename):
    all_seed_tunings, f = get_tuning_of_fft_of_V1id(T = 0.1, path = dirpath, V1id = V1id)
    np.save(dirpath + 'fq_amps_all_seed_tunings_%i.npy'%V1id, all_seed_tunings)

# the indices of dLGN cells that converge to V1 #V1id
idx = conn_LV[V1id]
ONidx = idx[idx<int(3071/2)]
OFFidx = idx[idx>=int(3071/2)]
# get the indices of the dLGN cells that their spikes will be plotted
plot_ONidx = ONidx[random.sample(range(len(ONidx)), 10)]
plot_OFFidx = OFFidx[random.sample(range(len(OFFidx)), 10)]
# the spikes of all presynaptic dLGN cells
inpt_spk = get_input_tc_spikes(V1_id=V1id, deg=60, folder=dirpath + 'seed_2/')
ts = np.concatenate(inpt_spk, axis=0)
# load the temporal tunings of the V1 neuron #V1id at all stimulus orientations, shape (30000, 12, 50)
all_seed_tunings = np.load(filename)
T = 0.1
freq = np.linspace(0., 1/(2*T)*1000., int(6000./T/2))


# plot and save figure

dpi = 300
width = 5.2
height = 0.82*width
lw=2.

fgsz = (width, height)
plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=fgsz)

gs = gridspec.GridSpec(2, 2, left=0.12, right=0.93, wspace=0.37, hspace=.4, bottom=0.12, top=0.95)

plot_raster_plot(gs[0,0], inpt_spk, plot_ONidx, plot_OFFidx)
plot_instantaneous_current(gs[0,1], ts, 5)
plot_new_power_spectrum_ylog(fig, gs[1,0], all_seed_tunings)
plot_new_fft_OSI(fig,gs[1,1])

fs = 10
font0  = FontProperties()
font0.set_weight('semibold')

fig.text(0.025, 0.96, 'A', fontsize=fs, fontproperties=font0)
fig.text(0.5, 0.96, 'B', fontsize=fs, fontproperties=font0)
fig.text(0.025, 0.47, 'C', fontsize=fs, fontproperties=font0)
fig.text(0.5, 0.47, 'D', fontsize=fs, fontproperties=font0)

savefig = False
if savefig == True:
    plt.savefig('figure_thalamic_fft.png', dpi=dpi)
    plt.savefig('figure_thalamic_fft.eps', dpi=dpi)
    plt.close()







