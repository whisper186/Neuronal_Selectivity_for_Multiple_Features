####### Wenqing Wei ###############################################
####### Plot the influence of convergence number from dLGN ########
####### to V1 neurons on their orientation selectivity. ###########
####### Network Simulation ########################################
####### The simulation runs on NEST 2.20 ##########################
###################################################################

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import random
import matplotlib.cm as cm
import math
from matplotlib import gridspec
import matplotlib.collections
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties


save_path = 'output_data/simu_convergence/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

##################################################################
# run simulation for different convergence number ################
# If the data is ready and only want to plot out  ################
# figures, comment out this part of code. ########################
##################################################################
import os
import parameters; reload(parameters); from parameters import *
import network_configuration; reload(network_configuration);
import network_configuration as network
import random_position_and_AmpPhases; reload(random_position_and_AmpPhases);
import random_position_and_AmpPhases as rpAP

n_lvs = np.array([2, 10, 20, 50, 100, 200, 300, 400, 500, 1000])
n_lps = np.array([1, 1, 2, 4, 8, 16, 24, 32, 40, 80])
g_FFIs = np.array([0.32, 1.6, 1.6, 2., 2., 2., 2., 2., 2., 2.])
As = 10000./n_lvs

# First create the profiles of the locations of RF center of dLGN neurons for different convergence number.
# Total number of dLGN neurons calculated from the convergence number.
N_LGNs = ((vf/rf)**2 * n_lvs).astype(int)  

loc_path = 'input_data/simu_convergence/'
if not os.path.exists(loc_path):
    os.makedirs(loc_path)

for i in range(len(N_LGNs)):
    LGN_locs = rpAP.get_random_points(N_LGNs[i], vf, np.array([0., 0.]))
    np.save(loc_path + 'LGN_locs/LGN_locs_sumN_%i.npy'%n_lvs[i], LGN_locs)


# Generate the amplitudes and phases of compound inputs to V1 neurons for different convergent number from LGN --> V1.
absAmps = np.empty((3, len(n_lvs)))
absAmps[0, :] = n_lvs
thalOSIs = np.empty((2, len(n_lvs)))
for i in range(len(n_lvs)):
    LGN_locs = np.load(loc_path + 'LGN_locs/LGN_locs_sumN_%i.npy'%n_lvs[i])
    amps, phases = rpAP.input_thalamic_compound_input(LGN_locs, network.V1_locs, A=1, Fre_spa = 0.08, sigc=1., sigs=5., n_lv = n_lvs[i])
    aOSI, aPO = rpAP.V1_OSI_PO(amps)
    absAmps[1, i] = amps.mean() * As[i]/2
    absAmps[2, i] = amps.std() * As[i]/2
    thalOSIs[0, i] = np.mean(aOSI)
    thalOSIs[1, i] = np.std(aOSI)
    np.save(save_path + 'convergence_amps_phases/amps_sumN_%i_1-5_1.npy'%(n_lvs[i]), amps)
    np.save(save_path + 'convergence_amps_phases/phases_sumN_%i_1-5_1.npy'%(n_lvs[i]), phases)
np.save(save_path + 'convergence_absolute_amp.npy', absAmps)
np.save(save_path + 'convergence_thaOSI_mean_std.npy', thalOSIs)


# run simulation and save simulation data
mnOSIs = np.empty((len(n_lvs), 2))
mnrates = np.empty((len(n_lvs), 2))

for i in range(len(n_lvs)):
    LGN_locs = np.load(loc_path + 'LGN_locs/LGN_locs_sumN_%i.npy'%n_lvs[i])
    print (n_lvs[i], len(LGN_locs))
    V1_rates = network.get_V1_neuron_rates(save_path, network.V1_locs, LGN_locs, network.FFI_locs, ext_rate, A=As[i], Fre_spa=0.08, sigc=1., sigs=5., J_LF=g_FFIs[i], J_FFI_V1=-1.6, n_LF=n_lps[i], n_LV=n_lvs[i], n_FFI_V1=320, cm=1.0, a=2, save_spk=False)
    np.save(loc_path + 'spk_rates_sumN_%i-A_%i.npy'%(n_lvs[i], As[i]), V1_rates)
    
    OSI, PO = rpAP.V1_OSI_PO(V1_rates)
    print ('mOSI : %.4f'%np.mean(OSI))
    mnOSIs[i, 0] = np.mean(OSI)
    mnOSIs[i, 1] = np.std(OSI)
    
    mnrates[i, 0] = np.mean(V1_rates)
    mnrates[i, 1] = np.std(V1_rates)
np.save(save_path + 'mOSI_std.npy', mnOSIs)
np.save(save_path + 'mrates_std.npy', mnrates)
print (np.array([n_lv, mnOSIs[:,0]]).T)



###################################################################
# plot functions ##################################################
###################################################################

lw=2.

def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_mean_error(gss, xdata, mean, error, ylabel, ylim, yticks):
    ax = plt.subplot(gss)
    
    ax.plot(xdata, mean, color='k', lw=lw, linestyle='solid', marker='.')
    ax.fill_between(xdata, mean-error, mean+error, facecolor='lightgray', alpha=1, edgecolor='none')
    #ax.vlines(x=100., ymin=0., ymax=mean[xdata==100], lw=lw, linestyle='dashed', color='k')
    ax.vlines(x=100., ymin=0., ymax=ylim[1]*0.8, lw=lw, linestyle='dashed', color='k')
    hide_axis(ax)
    ax.set_xscale('log')
    ax.set_ylabel(ylabel)
    ax.set_xlim(2., 1050)
    ax.set_ylim(ylim)
    ax.set_yticks(yticks)




abs_amp = np.load(save_path + 'convergence_absolute_amp.npy') # shape(3, 10) n_lvs, mean, std
LGN_OSI = np.load(save_path + 'convergence_thaOSI_mean_std.npy')  # shape(2, 10)
OSIs = np.load(save_path + 'mOSI_std.npy')  # shape(10,2)
rates = np.load(save_path + 'mrates_std.npy')  # shape(10,2)

conNr = abs_amp[0]


lw = 2.
dpi=300
plt.rcParams.update({'font.size': 8})
fs = 10
font0  = FontProperties()
font0.set_weight('semibold')


width=5.2
height=0.8*width

fgsz=(width, height)
fig = plt.figure(figsize=fgsz)

gs = gridspec.GridSpec(2, 2, left=0.11, right=0.95, wspace=.4, hspace=.25, bottom=0.13, top=0.94)

plot_mean_error(gs[0,1], conNr, OSIs[:,0], OSIs[:,1], ylabel=r'$\mathrm{OSI}_\mathrm{F0}^\mathrm{V1}$', ylim=(0., 0.6), yticks=[0.0, 0.2, 0.4, 0.6])
plot_mean_error(gs[0,0], conNr, LGN_OSI[0], LGN_OSI[1], ylabel=r'$\mathrm{OSI}_\mathrm{F1}^\mathrm{LGN}$', ylim=(0., 0.6), yticks=[0.0, 0.2, 0.4, 0.6])

plot_mean_error(gs[1,0], conNr, abs_amp[1,:]/1000, abs_amp[2,:]/1000, ylabel=r'$\mathrm{I}_\mathrm{F1}^\mathrm{LGN}$ [nA]', ylim=(0, 4.2), yticks=[0, 1, 2, 3])
plot_mean_error(gs[1,1], conNr, rates[:, 0], rates[:, 1], ylabel=r'$\nu_\mathrm{F0}^\mathrm{V1}$ [Hz]', ylim=(0., 65), yticks=[0, 20, 40])

fig.text(0.22, 0.025, 'Convergence')
fig.text(0.69, 0.025, 'Convergence')



fig.text(0.035, 0.95, 'A', fontsize=fs, fontproperties=font0)
fig.text(0.52, 0.95, 'B', fontsize=fs, fontproperties=font0)
fig.text(0.035, 0.49, 'C', fontsize=fs, fontproperties=font0)
fig.text(0.52, 0.49, 'D', fontsize=fs, fontproperties=font0)

fig.text(0.26, 0.96, 'Input', fontsize=fs, fontproperties=font0)
fig.text(0.715, 0.96, 'Output', fontsize=fs, fontproperties=font0)

savefig = False
if savefig == True:
    plt.savefig('figure_convergence.png', dpi=dpi)
    plt.savefig('figure_convergence.eps', dpi=dpi)
    plt.close()

