####### Wenqing Wei ###############################################
####### Plot the effects of spatial frequencies. ##################
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


save_path = 'output_data/simu_spatialFrequencies/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

##################################################################
# run simulation for different spatial frequencies ###############
# If the data is ready and only want to plot out  ################
# figures, comment out this part of code. ########################
##################################################################
import parameters; reload(parameters); from parameters import *
import network_configuration; reload(network_configuration);
import network_configuration as network
import random_position_and_AmpPhases; reload(random_position_and_AmpPhases);
import random_position_and_AmpPhases as rpAP

# Generate the amplitudes and phases of compound inputs to F1 and V1 neurons for different spatial frequencies.
sFs = np.array([0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4])

for i in range(len(sFs)):                             
    amps, phases = rpAP.input_thalamic_compound_input(network.dLGN_locs, network.V1_locs, A=1, Fre_spa = sFs[i], sigc = 1., sigs = 5., n_lv=100)
    FFI_amps, FFI_phases = rpAP.input_FFI_amp_phase(network.FFI_locs, network.dLGN_locs, A=1, Fre_spa = sFs[i], sigc = 1., sigs = 5., n_lp = 8)
    np.save(save_path + 'sFs_amps_phases/amps_%.3f_1-5_1.npy'%(sFs[i]), amps)
    np.save(save_path + 'sFs_amps_phases/phases_%.3f_1-5_1.npy'%(sFs[i]), phases)
    np.save(save_path + 'sFs_amps_phases/FFI_amps_%.3f_1-5_1.npy'%(sFs[i]), FFI_amps)
    np.save(save_path + 'sFs_amps_phases/FFI_phases_%.3f_1-5_1.npy'%(sFs[i]), FFI_phases)


# run simulation and save data

sFs = np.array([0.001, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4])

ampOSIs = np.empty((3, len(sFs)))  # (spatial frequencies, I_F1 of dLGN, mean OSI of V1 neurons)
ampOSIs[0] = sFs
msOSIs = np.empty((len(sFs), 2))
allPOs = np.empty((N, len(sFs)))
for i in range(len(sFs)):
    amps = np.load(save_path + 'sFs_amps_phases/amps_%.3f_1-5_1.npy'%(sFs[i]))
    phases = np.load(save_path + 'sFs_amps_phases/phases_%.3f_1-5_1.npy'%(sFs[i]))
    FFI_amps = np.load(save_path + 'sFs_amps_phases/FFI_amps_%.3f_1-5_1.npy'%(sFs[i]))
    FFI_phases = np.load(save_path + 'sFs_amps_phases/FFI_phases_%.3f_1-5_1.npy'%(sFs[i]))
    V1_rates = network.get_compound_V1_rates(amps, phases,FFI_amps, FFI_phases, ext_rate, save_spk=False)
    np.save(save_path + 'com_spk_rates_%.3f.npy'%sFs[i], V1_rates)
    
    OSI, PO = V1_OSI_PO(V1_rates)
    print ('mOSI : %.4f'%(np.mean(OSI)))
    msOSIs[i, 0] = np.mean(OSI)
    msOSIs[i, 1] = np.std(OSI)
    allPOs[:, i] = PO
    ampOSIs[1, i] = amps.mean() * A/2
ampOSIs[2] = msOSIs[:, 0]
np.save(save_path + 'mOSI_std.npy', msOSIs)
np.save(save_path + 'all_V1_POs.npy', allPOs)
np.save(save_path + 'sfs_mean_amp_OSI.npy', ampOSIs)
print (np.array([sFs, msOSIs[:,0]]).T)

# calculate circular correlation
import analysis_cc; reload(analysis_cc); import analysis_cc as Acc
#allPOs = np.load(save_path + 'all_V1_POs.npy')
circ_corr = []
for i in range(len(sFs)):
    cc = Acc.circular_correlation(allPOs[:,9], allPOs[:,i])
    circ_corr.append(cc)
np.save(save_path + 'sf_circ_corr_0.08.npy', circ_corr)


###################################################################
# plot functions ##################################################
###################################################################

lw = 2.


def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_correlation(gss, xdata, ydata, xlabel, ylabel):
    ax = plt.subplot(gss)
    
    ax.scatter(xdata[0::10], ydata[0::10], s=2, color='olive')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=-2)
    ax.set_xlim((-5, 185))
    ax.set_ylim((-5, 185))
    ax.set_xticks([0,90,180])
    ax.set_yticks([0,90,180])
    hide_axis(ax)
    

def plot_mean_error(gss, xdata, mean, error):
    ax = plt.subplot(gss)
    
    ax.plot(xdata, mean, color='k', lw=lw, linestyle='solid', marker='.')
    ax.fill_between(xdata, mean-error, mean+error, facecolor='lightgray', edgecolor='none')
    hide_axis(ax)
    ax.set_xlabel(r'Spatial frequency [cpd]')
    ax.set_ylabel(r'$\mathrm{OSI}_\mathrm{F0}^\mathrm{V1}$')
    ax.set_xlim(0., 0.41)
    ax.set_ylim(0., 0.37)
    ax.set_yticks([0., 0.1, 0.2, 0.3])
    ax.set_xticks([0., 0.1, 0.2, 0.3, 0.4])
    #plt.tight_layout()


def linear_absolute_amp_OSI(gss, xdata, ydata):
    ax = plt.subplot(gss)
    
    ax.plot(xdata, ydata, '.', color='darkcyan')
    sortamp = np.sort(xdata)
    sortOSI = [x for _,x in sorted(zip(xdata, ydata))]
    fit = np.polyfit(sortamp, sortOSI, 1)
    fit_fn = np.poly1d(fit)
    ax.plot(sortamp, fit_fn(sortamp), color='k', linestyle='dashed', lw=lw)
    ax.set_yticks([0., 0.1, 0.2])
    ax.set_xlim(0., 400.)
    ax.set_ylim(0., 0.27)
    ax.set_xlabel(r'$\mathrm{I}_\mathrm{F1}^\mathrm{LGN}$ [pA]')
    ax.set_ylabel(r'$\mathrm{OSI}_\mathrm{F0}^\mathrm{V1}$')
    hide_axis(ax)



sFs_OSIs = np.load(save_path + 'mOSI_std.npy')
sf_ccs = np.load(save_path + 'sf_circ_corr_0.08.npy')

amp_OSI = np.load(save_path + 'sfs_mean_amp_OSI.npy')
all_POs = np.load(save_path + 'all_V1_POs.npy')


dpi = 300

width=5.2
height=0.65*width

fgsz = (width, height)
plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=fgsz)


gs = gridspec.GridSpec(1, 2, left=0.12, right=0.95, wspace=.3, hspace=.6, bottom=0.6, top=0.95, width_ratios=[4, 6])


plot_mean_error(gs[0,0], sf_ccs[0], sFs_OSIs[:,0], sFs_OSIs[:,1])


ax2 = plt.subplot(gs[0,1:])
ax2.plot(sf_ccs[0], sf_ccs[1], marker='.', linestyle='solid', linewidth=lw, color='k', label='reference: 0.08 cpd')
ax2.set_yticks([0., 0.5, 1.0])
ax2.set_xticks([ 0.01, 0.08, 0.15,  0.2, 0.3, 0.4])
ax2.set_xlabel('Spatial frequency [cpd]')
ax2.set_ylabel('Circ corr. of PO')
ax2.legend(handlelength=0, markerscale=0, frameon=False, loc='upper right', bbox_to_anchor=(0.9, 0.85))
hide_axis(ax2)


gsb = gridspec.GridSpec(1, 3, left=0.12, right=0.95, wspace=.7, bottom=0.13, top=0.43)

linear_absolute_amp_OSI(gsb[0,0], amp_OSI[1], amp_OSI[2])

plot_correlation(gsb[0,1], all_POs[:,9], all_POs[:,8], xlabel=r'PO[$^\circ$](0.08 cpd)', ylabel=r'PO[$^\circ$](0.07 cpd)')
plot_correlation(gsb[0,2], all_POs[:,9], all_POs[:,2], xlabel=r'PO[$^\circ$](0.08 cpd)', ylabel=r'PO[$^\circ$](0.01 cpd)')


fs = 10
font0  = FontProperties()
font0.set_weight('semibold')

fig.text(0.02, 0.96, 'A', fontsize=fs, FontProperties=font0)
fig.text(0.43, 0.96, 'B', fontsize=fs, FontProperties=font0)
fig.text(0.02, 0.44, 'C', fontsize=fs, FontProperties=font0)
fig.text(0.36, 0.44, 'D', fontsize=fs, FontProperties=font0)
fig.text(0.68, 0.44, 'E', fontsize=fs, FontProperties=font0)


savefig = False
if savefig == True:
    plt.savefig('figure_spatialFrequencies.png', dpi=dpi)
    plt.savefig('figure_spatialFrequencies.eps', dpi=dpi)
    plt.close()


