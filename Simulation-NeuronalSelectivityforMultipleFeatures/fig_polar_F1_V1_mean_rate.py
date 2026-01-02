####### Wenqing Wei ###############################################
####### Plot the polar curves of input and output of ##############
####### some example neurons in the network. ######################
###################################################################


import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import random
import matplotlib.cm as cm
from matplotlib import gridspec
import matplotlib.collections
import matplotlib.colors as colors
from importlib import reload
import get_input_current; reload(get_input_current);
import get_input_current as InpCur


def plot_without_top_right_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plot_input_mean_F1(ax, mean, F1):
    #ax = plt.subplot(gss, polar=True)
    
    degree = np.arange(0., 360.+15, 15)
    listF1 = list(F1)*2 + [F1[0]]
    listmean = list(abs(mean))*2 + [abs(mean[0])]
    theta = degree * np.pi/180.
    
    ax.plot(theta, listF1, color='green', linewidth=lw, label='F1')
    ax.plot(theta, listmean, color='orange', linewidth=lw, label='F0')
    
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_ylim(0., 1257.)

def plot_output_mean_rates(ax, rates):
    #ax = plt.subplot(gss, polar=True)
    
    degree = np.arange(0., 360.+15, 15)
    theta = degree*np.pi/180.
    listrates = list(rates)*2 + [rates[0]]
    
    ax.plot(theta, listrates, color='purple', linewidth=lw)
    ax.xaxis.set_visible(True)
    ax.yaxis.set_visible(False)
    ax.spines['polar'].set_visible(False)
    ax.set_xticklabels([])
    ax.set_ylim(0., 41.)


def single_OSI(data):
    deg_range = np.arange(0., 180, 15)
    dsv = np.sum(data * np.exp(2*np.pi*1j * deg_range/180.))
    ndsv = np.sum(data)
    if ndsv == 0:
        dsv, ndsv = 0, 1
    dsvs = dsv/ndsv
    po = np.angle(dsvs)
    if po < 0:
        po = po + 2*np.pi
    po = po/(2*np.pi)*180
    osi = abs(dsvs)
    return osi, po



if not os.path.exists(folder + 'io_id_mean_F1_rates.npy'):
    InpCur.generate_data_for_figure()


lw = 2.
ids_mean_F1_rates = np.load(folder + 'io_id_mean_F1_rates.npy')
ids = ids_mean_F1_rates[0,0]
mean_F1_rates = ids_mean_F1_rates[1:,:,:]

dpi = 300

width = 7.5
height = 0.45*width

fgsz = (width, height)
plt.rcParams.update({'font.size': 8})

fig = plt.figure(figsize=fgsz)

gs = gridspec.GridSpec(2, 5, left=0.05, right=0.97, wspace=.1, hspace=0.2, bottom=0.03, top= 0.9)

for i in range(1, len(ids)):
    ax0 = plt.subplot(gs[0,i], polar=True)
    plot_input_mean_F1(ax0, mean_F1_rates[0,:,i], mean_F1_rates[1,:,i])
    ax1 = plt.subplot(gs[1,i], polar=True)
    plot_output_mean_rates(ax1, mean_F1_rates[2,:,i])

ax00=plt.subplot(gs[0,0], polar=True)
plot_input_mean_F1(ax00, mean_F1_rates[0,:,0], mean_F1_rates[1,:,0])
ax00.set_ylabel('Current [pA]')
leg = ax00.legend(loc='lower left', frameon=False, handlelength=0.5, bbox_to_anchor=(-0.05, -0.1))
leg.get_frame().set_alpha(0.5)
ax10=plt.subplot(gs[1,0], polar=True)
plot_output_mean_rates(ax10, mean_F1_rates[2,:,0])
ax10.set_ylabel(r'$\nu$ [Hz]')

fig.text(0.02, 0.67, r'I$^\mathrm{FF}$ [pA]', rotation=90)
fig.text(0.02, 0.18, r'$\nu^\mathrm{V1}$ [Hz]', rotation=90)

fig.text(0.1, 0.93 , '# %i'%ids[0])
fig.text(0.29, 0.93 , '# %i'%ids[1])
fig.text(0.47, 0.93 , '# %i'%ids[2])
fig.text(0.67, 0.93 , '# %i'%ids[3])
fig.text(0.85, 0.93 , '# %i'%ids[4])

savefig = False
if savefig == True:
    plt.savefig('figure_io_polar_curve.png', dpi=dpi)
    plt.savefig('figure_io_polar_curve.eps', dpi=dpi)




