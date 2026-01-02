####### Wenqing Wei ###############################################
####### Plot the effects of stimulus contrasts. ###################
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


save_path = 'output_data/simu_contrast/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

##################################################################
# run simulation for different stimulus contrasts ################
# If the data is ready and only want to plot out  ################
# figures, comment out this part of code. ########################
##################################################################
import network_configuration; reload(network_configuration);
import network_configuration as network

############################################################
# run for different contrasts ##############################
############################################################

Fre_tem = 3.
cms = np.array([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
cm_OSIs = np.empty((N, len(cms)))
cm_POs = np.empty((N, len(cms)))
for i in range(len(cms)):
    V1_rates = network.get_V1_neuron_rates(save_path, network.V1_locs, network.dLGN_locs, network.FFI_locs, network.ext_rate, cm = cms[i], a=2, save_spk=False)
    np.save(save_path + 'com_spk_rates_cm_%.1f.npy'%cms[i], V1_rates)
    
    OSI, PO = V1_OSI_PO(V1_rates)
    cm_POs[:,i] = PO
    cm_OSIs[:, i] = OSI
    print ('mOSI : %.4f'%np.mean(OSI))
np.save(save_path + 'cm_OSIs.npy', cm_OSIs)
np.save(save_path + 'cm_POs.npy', cm_POs)
print (np.array([cms, mcOSIs[:,0]]).T)

# extract mean and std of neurons' OSI at different contrasts.
msOSIs = np.empty((3, 6))
msOSIs[0] = cms
msOSIs[1] = cm_OSIs.mean(axis = 0)
msOSIs[2] = cm_OSIs.std(axis = 0)
np.save(save_path + 'cms_mean_std_OSI.npy', msOSIs)

# calculate circular differences between POs
import analysis_cc; reload(analysis_cc); import analysis_cc as Acc
allPOs = np.load(save_path + 'cm_POs.npy')
dPO = np.empty((12500, 6))
for i in range(1, len(cms)):
    degdiff = Acc.calculate_circular_difference(allPOs[:,0], allPOs[:,i])
    dPO[:,i] = degdiff
dPO[:,0] = 0.
np.save(save_path + 'cms_pair_diff_PO.npy', dPO)

# extract tuning curves of two neurons
V1id = [3013, 11335]
V1id_tc = np.empty((len(V1id), len(cms), len(network.deg_range)))
for i in range(len(cms)):
    V1rates = np.load(save_path + 'com_spk_rates_cm_%.1f.npy'%cms[i])
    V1id_tc[0, i, :] = V1rates[V1id[0]]
    V1id_tc[1, i, :] = V1rates[V1id[1]]
np.save(save_path + 'cm_rates_%i.npy'%(int(V1id[0])), V1id_tc[0])
np.save(save_path + 'cm_rates_%i.npy'%(int(V1id[1])), V1id_tc[1])

###################################################################
# plot functions ##################################################
###################################################################
lw = 2.
lp = 0.


def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_contrast_OSI(gss, xdata, mean, error):
    ax = plt.subplot(gss)
    
    ax.plot(xdata, mean, color='k', lw=lw, linestyle='solid', marker='.')
    ax.fill_between(xdata, mean-error, mean+error, facecolor='lightgray', alpha=1., edgecolor='none')
    hide_axis(ax)
    ax.set_xlabel('Contrast')
    ax.set_ylabel(r'$\mathrm{OSI}^{\mathrm{V1}}_{\mathrm{F0}}$')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.01, 0.35)
    ax.set_yticks([0., 0.1, 0.2, 0.3])



def plot_contrast_dPO(gss, xdata, mean, error):
    ax = plt.subplot(gss)
    
    ax.plot(xdata, mean, color='k', lw=lw, linestyle='solid', marker='.')
    ax.fill_between(xdata, mean-error, mean+error, facecolor='lightgray', alpha=1., edgecolor='none')
    hide_axis(ax)
    ax.set_xlabel('Contrast')
    ax.set_ylabel(r'Deviation in PO [$^\circ$]')
    ax.set_yticks([0, 30, 60, 90])
    ax.set_xlim(-0.02, 1.02)



colors = ['purple', 'darkmagenta', 'orchid', 'plum', 'thistle']

def plot_contrast_tc(gss, cm_rates, leg=False): 
    
    ax1 = plt.subplot(gss)    
    
    deg_range = np.arange(0., 180.+15, 15)
    cms = np.array([1.0, 0.8, 0.5, 0.3, 0.1, 0.0])
    for i in range(5):
        ax1.plot(deg_range, np.append(cm_rates[i], cm_rates[i, 0]), color=colors[i], linewidth=lw, label='cm=%.1f'%cms[i])#, alpha=cms[i])
    ax1.set_rasterized(True)
    ax1.plot(deg_range, np.append(cm_rates[-1], cm_rates[-1, 0]), color='k', linestyle='dotted', linewidth=lw, label='cm=0.0')
    hide_axis(ax1)
    ax1.set_xlabel(r'$\theta [^\circ]$')#, labelpad=-1)
    ax1.set_ylabel(r'$\nu$ [Hz]')
    ax1.set_xlim(0., 185.)
    ax1.set_xticks([0., 60., 120., 180.])
    ax1.set_ylim(bottom=0.)
    if leg==True:
        ax1.legend(bbox_to_anchor=(0.5, 1.15))

# load data
cm_OSI = np.load(save_path + 'cms_mean_std_OSI.npy') # shape(3, 6) cms, mean, std
cm_dPO = np.load(save_path + 'cms_pair_diff_PO.npy')  # shape(12500, 6)

V1id = [3013, 11335]
tunings_0 = np.load(save_path + 'cm_rates_%i.npy'%(int(V1id[0])))
tunings_1 = np.load(save_path + 'cm_rates_%i.npy'%(int(V1id[1])))

xdata = cm_OSI[0]

dPO_mean = cm_dPO.mean(axis=0)
dPO_std = cm_dPO.std(axis=0)

# plot and save figure

lw=3.
dpi=300
width = 5.2
height = 0.76*width

fgsz = (width, height)
plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=fgsz)

gs = gridspec.GridSpec(2,2, left=0.12, right=0.95, wspace=.35, hspace=.42, bottom=0.1, top=0.96)
plot_contrast_OSI(gs[0,0], xdata, cm_OSI[1], cm_OSI[2])
plot_contrast_dPO(gs[0,1], xdata, dPO_mean, dPO_std)
plot_contrast_tc(gs[1,0], tunings_0)
plot_contrast_tc(gs[1,1], tunings_1, leg=True)


fs = 10
font0  = FontProperties()
font0.set_weight('semibold')

fig.text(0.03, 0.96, 'A', fontsize=fs, fontproperties=font0)
fig.text(0.52, 0.96, 'B', fontsize=fs, fontproperties=font0)
fig.text(0.03, 0.45, 'C', fontsize=fs, fontproperties=font0)
fig.text(0.52, 0.45, 'D', fontsize=fs, fontproperties=font0)

savefig = False
if savefig == True:
    plt.savefig('figure_contrast.png', dpi=dpi)
    plt.savefig('figure_contrast.eps', dpi=dpi)
    plt.close()
