####### Wenqing Wei ###############################################
####### Plot distribution of the orientation selectivity ##########
####### of V1 neurons and their dLGN inputs. ######################
###################################################################

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import random
import matplotlib.cm as cm
import math
import os
from matplotlib import gridspec
import matplotlib.collections
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties

from importlib import reload
import get_input_current; reload(get_input_current);
import get_input_current as InpCur



##########################################################
# load/generate data #####################################
##########################################################

filenames = [InpCur.folder + 'PO_corr_spk.npy', InpCur.folder + 'OSI_corr_spk.npy']

if not all(os.path.exists(f) for f in filenames):
    print('At least one file is missing')
    print('Generating data ...')

    inputF1s = np.empty((12500, len(deg_range)))
    for i in range(inputF1s.shape[0]):
        means, F1s = InpCur.get_input_mean_F1(i)
        inputF1s[i, :] = F1s
    np.save(InpCur.folder + 'inputF1s_corr_spk.npy', inputF1s)

    inOSI, inPO = InpCur.V1_OSI_PO(inputF1s)
    V1_rates = np.load(path + 'V1_rates')
    opOSI, opPO = InpCur.V1_OSI_PO(V1_rates)
    io_POs = np.array([inPO, opPO])
    io_OSIs = np.array([inOSI, opOSI])
    np.save(InpCur.folder + 'PO_corr_spk.npy', io_POs)
    np.save(InpCur.folder + 'OSI_corr_spk.npy', io_OSIs)
    print('Finish generating data!')


io_POs = np.load(InpCur.folder + 'PO_corr_spk.npy')
io_OSI = np.load(InpCur.folder + 'OSI_corr_spk.npy')




##########################################################
# plotting ###############################################
##########################################################

def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_hist_OSI(gss, OSIs, xlabel):
    
    ax = plt.subplot(gss)
    
    height, bins, patches = plt.hist(OSIs, weights=np.ones(len(OSIs))/len(OSIs), bins=np.linspace(0., 1.0, 20), edgecolor='k', color='silver', label='mean=%.2f'%np.mean(OSIs))
    ax.set_xlim(0., 1.0)
    ax.set_ylim(bottom=0.)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fraction')
    ax.text(0.8,0.8, 'mean = %.2f'%np.mean(OSIs), ha="right", va="top", transform=plt.gca().transAxes)
    hide_axis(ax)

def plot_io_corr(gss, xdata, ydata, xlabel, ylabel, lim, ticks):
    
    ax = plt.subplot(gss)
    
    ax.scatter(xdata[::5], ydata[::5], s=3, color='darkcyan', alpha=0.3)
    ax.plot((0., 0.5), (0., 0.5), 'k--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    hide_axis(ax)


def plot_2d_hist_OSI(gss, LGN_OSI, V1_OSI):
    ax = plt.subplot(gss)

    ybin=np.linspace(0., 0.6, 50)
    
    counts, xedge, yedge, im = ax.hist2d(LGN_OSI, V1_OSI, bins=(ybin, ybin), cmap=plt.cm.BuPu)
    ax.set_xlabel(r'OSI$_\mathrm{F1}^\mathrm{LGN}$')
    ax.set_ylabel(r'OSI$_\mathrm{F0}^\mathrm{V1}$')
    ax.set_xlim(0., 0.62)
    ax.set_ylim(0., 0.62)
    ax.set_xticks([0., 0.2, 0.4, 0.6])
    ax.set_yticks([0., 0.2, 0.4, 0.6])
    ax.plot((0., 0.6), (0., 0.6), color='k')
    cbar = plt.colorbar(im)
    cbar.set_ticks([0, 20, 40])
    cbar.set_label('# V1 neurons')
    hide_axis(ax)


dpi = 300
width = 5.2
height=0.82*width

fgsz = (width, height)
plt.rcParams.update({'font.size':8})

fig = plt.figure(figsize=fgsz)
gs = gridspec.GridSpec(2, 2, left=0.12, right=0.93, wspace=.37, hspace=.4, bottom=0.12, top=0.95)

plot_hist_OSI(gs[0,1], io_OSI[1], xlabel=r'$\mathrm{OSI}^{\mathrm{V1}}_{\mathrm{F0}}$')
plot_hist_OSI(gs[0,0], io_OSI[0], xlabel=r'$\mathrm{OSI}_\mathrm{F1}^\mathrm{LGN}$')

plot_io_corr(gs[1,0], io_POs[0], io_POs[1], xlabel=r'$\mathrm{PO}_\mathrm{F1}^\mathrm{LGN} \ [^\circ]$', ylabel=r'$\mathrm{PO}_\mathrm{F0}^\mathrm{V1} \ [^\circ]$', lim=(-5, 185), ticks=[0,90, 180])
plot_2d_hist_OSI(gs[1,1], io_OSI[0], io_OSI[1])

fs = 10
font0  = FontProperties()
font0.set_weight('semibold')

fig.text(0.025, 0.96, 'A', fontsize=fs, fontproperties=font0)
fig.text(0.5, 0.96, 'B', fontsize=fs, fontproperties=font0)
fig.text(0.025, 0.47, 'C', fontsize=fs, fontproperties=font0)
fig.text(0.5, 0.47, 'D', fontsize=fs, fontproperties=font0)


savefig = False
if savefig == True:
    plt.savefig('figure_OS_io_distribution.png', dpi=dpi)
    plt.savefig('figure_OS_io_distribution.eps', dpi=dpi)

