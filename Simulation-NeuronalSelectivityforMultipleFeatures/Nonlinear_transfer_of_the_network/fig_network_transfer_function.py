############ Wenqing Wei ###################
############ plot figures 10 and 11 for the transfer function of network
####### Wenqing Wei ###############################################
####### Plot the nonlinear transfer of the network and compare ####
####### the analytical results with numerical simulation. #########
###################################################################

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import matplotlib.cm as cm
import math

from matplotlib import gridspec
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
from importlib import reload

def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].axis.axes.tick_params(direction="in")


save_path = 'output_data/'
if not os.path.exists(save_path):
    print ('No output data! You should first run the simulation!')
    import fullSolution; reload(fullSolution)
    import analysis; reload(analysis)

ff_current = np.load(save_path + 'ff_input_current.npy')
all_current = np.load(save_path + 'all_input_current.npy')
rate_model = np.load(save_path + 'full_rates_nt60.npy')

ta = np.linspace(0., 0.3, 60)
plotdeg = [105, 75]
V1id = 3013

# 105 degree
I_ff = ff_current[int(plotdeg[0]/15), V1id, :]
I_all = all_current[int(plotdeg[0]/15), V1id, :]
rate_all = rate_model[int(plotdeg[0]/15), V1id, :]

# 75 degree
I_ff2 = ff_current[int(plotdeg[1]/15), V1id, :]
I_all2 = all_current[int(plotdeg[1]/15), V1id, :]
rate_all2 = rate_model[int(plotdeg[1]/15), V1id, :]

lw = 2
dpi = 300
color_cycle = ['green', 'crimson', 'purple', 'darkseagreen', 'lightcoral', 'plum']

fs = 10
font0 = FontProperties()
font0.set_weight('semibold')

width = 5.2
height = 0.82*width
fgsz = (width, height)
plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=fgsz)
gs = gridspec.GridSpec(3,3, left=0.09, right=0.97, wspace=0., hspace=0., bottom=0.12, top=0.95, width_ratios = [5,1,6], height_ratios = [5,1,4])

xml = (-0.01, 0.31)
xil = (-1950, 700)
yl = (-1, 101)

ax_outdeg = plt.subplot(gs[0,0])
ax_outdeg.plot(ta, rate_all, color=color_cycle[2], linewidth=lw, label='105$^\circ$')
ax_outdeg.hlines(y=np.mean(rate_all), xmin=-0.01, xmax=0.33, colors=color_cycle[2], linestyle='solid', linewidth=lw-1)
ax_outdeg.plot(ta, rate_all2, color=color_cycle[5], linewidth=lw, label='75$^\circ$')
ax_outdeg.hlines(y=np.mean(rate_all2), xmin=-0.01, xmax=0.33, colors=color_cycle[5], linestyle='solid', linewidth=lw-1)
ax_outdeg.set_ylim(yl)
ax_outdeg.set_xlim(xml)
ax_outdeg.set_xticks([0., 0.1, 0.2, 0.3])
ax_outdeg.set_xlabel('t [s]', labelpad=-1)
ax_outdeg.set_ylabel(r'$\nu$ [Hz]')
ax_outdeg.legend(loc='upper right', frameon=False)
hide_axis(ax_outdeg)

ax_outdeg1 = plt.subplot(gs[0,1])
ax_outdeg1.axis('off')
ax_outdeg1.hlines(y=np.mean(rate_all), xmin=-0.1, xmax=1.1, colors=color_cycle[2], linestyle='solid', linewidth=lw-1)
ax_outdeg1.hlines(y=np.mean(rate_all2), xmin=-0.1, xmax=1.1, colors=color_cycle[5], linestyle='solid', linewidth=lw-1)
ax_outdeg1.set_ylim(yl)
ax_outdeg1.set_xlim(xml)

ax_maindeg = plt.subplot(gs[0,2])
ax_maindeg.scatter(ff_current[0,:1000,:].flatten(), rate_model[0,:1000,:].flatten(), s=3, color=color_cycle[0], label=r'$\mathrm{I}_\mathrm{ff}$', clip_on=False)
ax_maindeg.scatter(all_current[0,:1000,:].flatten(), rate_model[0,:1000,:].flatten(), s=3, color=color_cycle[1], label=r'$\mathrm{I}_\mathrm{all}$', clip_on=False)
ax_maindeg.hlines(y=np.mean(rate_all), xmin=-1950., xmax=-200, colors=color_cycle[2], linestyle='solid', linewidth=lw-1)
ax_maindeg.hlines(y=np.mean(rate_all2), xmin=-1950., xmax=-320, colors=color_cycle[5], linestyle='solid', linewidth=lw-1)

ax_maindeg.vlines(x=np.mean(I_ff), ymin=-1, ymax=1., colors=color_cycle[0], linestyle='solid', linewidth=lw-1)
ax_maindeg.vlines(x=np.mean(I_all), ymin=-1, ymax=1.5, colors=color_cycle[1], linestyle='solid', linewidth=lw-1)

ax_maindeg.set_ylim(yl)
ax_maindeg.set_xlim(xil)
ax_maindeg.set_xticks([-1000, -500, 0, 500])
ax_maindeg.set_yticks([0., 0.1, 0.2, 0.3])
ax_maindeg.set_xlabel('Current [pA]')
ax_maindeg.set_ylabel(r'$\nu$ [Hz]')
ax_maindeg.spines['bottom'].axis.axes.tick_params(direction='in')
ax_maindeg.legend(frameon=False, loc='upper left')
ax_maindeg.axis('off')

ax_indeg = plt.subplot(gs[2,2])
ax_indeg.plot(I_ff, ta, color=color_cycle[0], linewidth=lw, label=r'$\mathrm{I}_\mathrm{ff}$')
ax_indeg.vlines(x=np.mean(I_ff), ymin=-1, ymax=5, linestyle='solid', color=color_cycle[0], linewidth=lw-1)
ax_indeg.plot(I_ff2, ta, color=color_cycle[3], linewidth=lw)
ax_indeg.plot(I_all, ta, color=color_cycle[1], linewidth=lw, label=r'$\mathrm{I}_\mathrm{all}$')
ax_indeg.vlines(x=np.mean(I_all), ymin=-1, ymax=5, linestyle='solid', color=color_cycle[1], linewidth=lw-1)
ax_indeg.plot(I_all2, ta, color=color_cycle[4], linewidth=lw)
ax_indeg.set_xlabel('Current [pA]')
ax_indeg.set_ylabel('t [s]')
ax_indeg.set_ylim(xml)
ax_indeg.set_yticks([0., 0.1, 0.2, 0.3])
ax_indeg.set_xlim(xil)
hide_axis(ax_indeg)


ax_indeg1 = plt.subplot(gs[1,2])
ax_indeg1.axis('off')
ax_indeg1.vlines(x=np.mean(I_ff), ymin=-1, ymax=5, colors=color_cycle[0], linestyle='solid', linewidth=lw-1)
ax_indeg1.vlines(x=np.mean(I_all), ymin=-1, ymax=5, colors=color_cycle[1], linestyle='solid', linewidth=lw-1)
ax_indeg1.set_xlim(xil)
ax_indeg1.set_ylim(0,1)

fig.text(0.03, 0.96, 'A', fontsize=fs, FontProperties=font0)
fig.text(0.47, 0.96, 'B', fontsize=fs, FontProperties=font0)
fig.text(0.47, 0.47, 'C', fontsize=fs, FontProperties=font0)

savefig1 = False
if savefig1 == True:
    plt.savefig('figure_netTransFunc.png', dpi=dpi)
    plt.savefig('figure_netTransFunc.eps', dpi=dpi)
    plt.close()



com_OSI_PO = np.load(save_path + 'num_com_V1_OSI_PO.npy')
transFunc_OSI_PO = np.load(save_path + 'transFunc_OSI_PO.npy')


def plot_correlation(gs, xdata, ydata, xlabel, ylabel, xlim, ylim, ticks):
    ax = plt.subplot(gs)
    ax.scatter(xdata[0::5], ydata[0::5], s=2, color='olive')
    ax.scatter(xdata, ydata, s=2, color='olive')
    diagonal = np.max(xdata)#*0.9
    ax.plot((0.,diagonal), (0., diagonal), color='silver')
    ax.set_xlabel(xlabel)#, labelpad=-0.5)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    hide_axis(ax)



width=5.2
height=0.48*width

fgsz = (width, height)
fig = plt.figure(figsize=fgsz)
gs = gridspec.GridSpec(1,2, left=0.12, right=0.97, wspace=.35, bottom=0.2, top=0.9)

plot_correlation(gs[0,0], com_OSI_PO[:,1], transFunc_OSI_PO[:,1], ylabel=r'PO$_{\mathrm{DRM}}$', xlabel=r'PO$_{\mathrm{SIM}}$', ylim=(-5, 185), xlim=(-5, 185), ticks=[0,90,180])
plot_correlation(gs[0,1], com_OSI_PO[:,0], transFunc_OSI_PO[:,0], ylabel=r'OSI$_{\mathrm{DRM}}$', xlabel=r'OSI$_{\mathrm{SIM}}$', ylim=(-0.02, 0.62), xlim=(-0.02, 0.62), ticks=[0., 0.2, 0.4, 0.6])

fig.text(0.04, 0.93, 'A', fontsize=fs, FontProperties=font0)
fig.text(0.52, 0.93, 'B', fontsize=fs, FontProperties=font0)


folder = '/Users/wenqingwei/Desktop/safety_margin_simulation/figures_in_paper/figure_10-11/'
plt.savefig(folder + 'figure_11_compare_model.png', dpi=dpi)
plt.savefig(folder + 'figure_11_compare_model.eps', dpi=dpi)

savefig2 = False
if savefig2 == True:
    plt.savefig('figure_compareModel.png', dpi=dpi)
    plt.savefig('figure_compareModel.eps', dpi=dpi)
    plt.close()
