####### Wenqing Wei #######################################
####### transfer function of a single LIF neuron ##########
###########################################################


import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import time
import os
from matplotlib import gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties

from scipy.special import erfcx
import scipy.integrate
import scipy.sparse


import nest

nest.set_verbosity('M_FATAL')
nest.ResetKernel()
nest.SetKernelStatus({'overwrite_files': True})
nest.SetKernelStatus({'print_time': True})


def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].axis.axes.tick_params(direction="in")

##############################################
## analytical solution #######################
##############################################
def cos_lif_rate(A0, A1, B, f_tf):
    '''Calculat the analytical estimation of the output rate with the given input. If static rate model is applied, f_tf = 1; if dynamic rate model is applied, f_tf is the attenuated factor depends on the temporal frequency.
    Parameters:
    A0: the mean value of the excitatory input.
    A1: the amplitude of the excitatory input.
    B: the mean value of the inhibitory input.
    f_tf: the attenuated factor.
    '''
    exc = A0 + f_tf*A1 * np.cos(2*np.pi*Fre_tem*t)
    inh = B 
    mu = (tau_m) * exc * J - (tau_m) * inh * J

    sig2 = (tau_m) * exc * J**2 + (tau_m) * inh * J**2

    sig = np.sqrt(sig2)
        
    low_lim = (V_r - mu)/sig
    up_lim = (V_th - mu)/sig
    
    integral = np.empty((mu.size))
    for i in range(len(mu)):
        integral[i] = scipy.integrate.quad(lambda x: erfcx(-x), low_lim[i], up_lim[i])[0]
    
    Phi = tau_ref + tau_m * np.sqrt(np.pi) * integral
    
    inpt = mu
    oupt = 1./Phi
    #print(time.time() - t0)
    return inpt, oupt  

##############################################
## numerical simulation ######################
##############################################
def cos_LIF_simu(A0, A1, B, Fre_tem, a):
    '''A single LIF neuron receives inputs from an excitatory input with temporally sinusoidal modulation and a constant inhibitory input.
    Parameters:
    A0: the mean value of the excitatory input.
    A1: the amplitude of the excitatory input.
    B: the mean value of the inhibitory input.
    Fre_tem: the temporal frequency.
    a: the random seed.'''
    
    nest.ResetKernel()
    
    nest.SetKernelStatus({'overwrite_files': True})
    nest.SetKernelStatus({'print_time': True})
    
    # set seed ====================================================================
    #a = int(theta*180/np.pi)
    msd = 1234 * a * 100 + 1
    n_vp_g = nest.GetKernelStatus('total_num_virtual_procs')
    msdrange2 = range(msd + n_vp_g + 1, msd + 2 * n_vp_g + 1)
    nest.SetKernelStatus({'rng_seeds':msdrange2,'grng_seed':msd + n_vp_g})
    #============================================================================== 
    
    
    lif = nest.Create('iaf_psc_delta', 1, { 'tau_m': 20.,
                                            't_ref': 2.,
                                            'V_th': 20.,
                                            'V_reset': 0.,
                                            'E_L': 0.})
    
    input_pg = nest.Create('sinusoidal_poisson_generator', 1, params={'frequency': Fre_tem, 'rate': A0, 'amplitude': A1, 'phase': 90.})
    inh_pg = nest.Create('poisson_generator', 1, params={'rate':B})
    
    syn_dict = {'model': 'static_synapse', 'weight': J}
    nest.Connect(input_pg, lif, 'one_to_one', syn_spec = syn_dict)
    
    syn_inh = {'model': 'static_synapse', 'weight': -J}
    nest.Connect(inh_pg, lif, 'one_to_one', syn_spec = syn_inh)
    
    sd = nest.Create('spike_detector', 1, {'start': 200.})
    
    nest.Connect(lif, sd)
    
    T = 600000.
    nest.Simulate(T + 200.)
    dSD = nest.GetStatus(sd, keys = 'events')[0]
    evs = dSD['senders']
    ts = dSD['times']
    return len(ts)*1000./T


###################################################################################

# The average rates: r_lgn = 83.9 Hz, r_pv = 36 Hz, r_V1 = 6.52, r_ext = 8350 Hz
# The synaptic weight:J_lgn = 2.0 mV, J_pv = -1.8 mV, J_V1ext = 0.2 mV, J_V1inh = -8*0.2 mV, J_ext = 0.1 mV
# Projection number: C_lgn = 100, C_pv = 320, C_V1ext = 1000, C_V1inh = 250, C_ext = 1

# mu_lgn = tau_m*r_lgn*J_lgn*C_lgn = 20ms * 83.9Hz * 2mV * 100 = 335.6 mV
# sig_lgn = np.sqrt(tau_m*r_lgn*J_lgn**2*C_lgn) = np.sqrt(20ms*83.9*4mV**2*100) = 25.9 mV

# mu_pv = tau_m*r_pv*J_pv*C_pv = 20*1e-3s * 36Hz * -1.8mV * 320 = -414.72 mV
# sig_pv = np.sqrt(20*1e-3s * 36Hz * 1.8**2 * 320) = 27.32 mV**2

# mu_exc = 20ms * 6.5Hz * 0.2mV * 1000 = 26 mV
# sig_exc = np.sqrt(20ms * 6.5Hz * 0.2**2 * 1000) = 2.28 mV**2


# mu_inh = 20ms * 6.5Hz * -1.6mV * 250 = -52 mV
# sig_inh = np.sqrt(20ms * 6.5Hz * 1.6**2 * 250) = 9.12 mV**2

# mu_ext = 20ms * 8350Hz * 0.1mV * 1 = 16.7 mV
# sig_ext = np.sqrt(20ms * 8350Hz * 0.1**2 * 1) = 1.29 mV**2

# mu_exc = mu_lgn + mu_exc + mu_ext = 378.3 mV
# mu_inh = mu_pv + mu_inh = -466.72 mV
# sig_exc = sig_lgn + sig_exc + sig_ext = 29.47 mV
# sig_inh = sig_pv + sig_inh = 36.44 mV

#NOTE: it is only a rough calculation of the input in order to extract the value of the variables of the single LIF that approximate the simulated network. 

# When Nu_exc = 20000 Hz and J = 1., then mu_exc = 400 mV and sig_exc = 20 mV^2. 
# When Nu_inh = 23336 Hz and J = -1., then mu_inh = -466.72 mV and sig_inh = 21.6 mV^2

###################################################################################


# Parameters

tau_m = 20.*1e-3        # ms, time constant
tau_ref = 2.*1e-3       # ms, refractory period

V_r = 0.0                 # mV, resting potential
V_th = 20.0               # mV, threshold voltage

C_m = 250.                # pF, membrane capacitance
R = tau_m/C_m * 1e3       # Resistance, M
# record input mu and conver to current. tau_m = R*C, C = 250pF, R = 20ms/250pF = 80M, I = V/R = mV/80M = 12.5 pA
facVtoI = 1 / R     # pA

a = 2   # the random seed of numerical simulation

J = 1.              # mV, synaptic weight from LGN --> V1
t = np.linspace(0., 1., 1000)   # s, the array time points
Fre_tem = 3.        # Hz, the temporal frequency
tfs = np.array([0.3,  0.6,  1. ,  2. ,  3. ,  6. , 10. , 20. , 30.])

A0 = 20000.     # Hz, the mean value of the excitatory input
A1 = 3500.      # Hz, the amplitude of the excitatory input
B = 23336.      # HZ, the mean of the inhibitory input

def single_LIF_transfer_function(t, Fre_tem, A0, A1, B):
    '''Extract the data for fig2 A1-A3.'''
    sLIF_data = np.empty((len(t), 4)) 
    sLIF_data[:, 0] = t
    att_fac = 1/np.sqrt(1+(2*np.pi*tau_m*Fre_tem)**2)
    ai, ao = cos_lif_rate(A0, A1, B, att_fac)
    i, o = cos_lif_rate(A0, A1, B, 1.)
    sLIF_data[:, 1] = i * facVtoI
    sLIF_data[:, 2] = ao
    sLIF_data[:, 3] = o
    
    return sLIF_data


def single_LIF_different_models(tfs, A0, A1, B):
    '''Extract the data for fig2 B.'''
    ana_rates = np.empty((len(tfs)))
    num_rates = np.empty((len(tfs)))
    attFacs = 1/np.sqrt(1+(2*np.pi*tau_m*tfs)**2)
    for i in range(len(tfs)):
        ai, ao = cos_lif_rate(A0, A1, B, attFacs[i])
        ana_rates[i] = np.mean(ao)
        
        r = cos_LIF_simu(A0, A1, B, tfs[i], a)
        num_rates[i] = r
    
    tf_data = np.empty((3, len(tfs)))
    tf_data[0] = tfs
    tf_data[1] = ana_rates
    tf_data[2] = num_rates
    
    return tf_data


def single_LIF_change_mean(A0s, A1, B, Fre_tem):
    sLIF_A0s = np.empty((len(A0s), len(t), 2))
    #attFac = 1/np.sqrt(1+(2*np.pi*tau_m*Fre_tem)**2)
    for i in range(len(A0s)):
        ai, ao = cos_lif_rate(A0s[i], A1, B, 1.)
        sLIF_A0s[i, :, 0] = ai * facVtoI
        sLIF_A0s[i, :, 1] = ao

    return sLIF_A0s

def single_LIF_change_amplitude(A0, A1s, B, Fre_tem):
    sLIF_A1s = np.empty((len(A1s), len(t), 2))
    #attFac = 1/np.sqrt(1+(2*np.pi*tau_m*Fre_tem)**2)
    for i in range(len(A1s)):
        ai, ao = cos_lif_rate(A0, A1s[i], B, 1.)
        sLIF_A1s[i, :, 0] = ai * facVtoI
        sLIF_A1s[i, :, 1] = ao
    return sLIF_A1s
        

def run_simulation_and_save_data(save = False):
    # fig2 A1-A3, the transfer function of single LIF cell.
    sLIF_data = single_LIF_transfer_function(t, Fre_tem, A0, A1, B)
    # fig2 B, the output rates of given input at different temporal frequencies.
    tf_data = single_LIF_different_models(tfs, A0, A1, B)
    # fig3 A-C, change the mean.
    A0s = np.arange(16800., 23400., 100)
    sLIF_A0s = single_LIF_change_mean(A0s, A1, B, Fre_tem)
    # fig3 D-F, change the amplitude.
    A1s = np.arange(0.,9100., 100)
    sLIF_A1s = single_LIF_change_amplitude(A0, A1s, B, Fre_tem)
    
    dict_data = {'sLIF_data': sLIF_data,
                 'tf_data': tf_data,
                 'sLIF_A0s': sLIF_A0s,
                 'sLIF_A1s': sLIF_A1s}
    if save == True:
        np.save('output_data/fig2-3_single_LIF_tf.npy', np.array([dict_data]))



if not os.path.exists('output_data/fig2-3_single_LIF_tf.npy'):
    run_simulation_and_save_data(save=True)
else:
    print ('file already exist.')



#############################################################
# plot figure 2 and 3 #######################################
#############################################################

all_data = np.load('output_data/fig2-3_single_LIF_tf.npy', allow_pickle=True)[0]
sLIF_data = all_data['sLIF_data']
tf_data = all_data['tf_data']
sLIF_A0s = all_data['sLIF_A0s']
sLIF_A1s = all_data['sLIF_A1s']

lw = 2
dpi = 300
color_cycle=['green', 'crimson', 'purple']

fs = 10
font0 = FontProperties()
font0.set_weight('semibold')

# figure 2
plotfig2 = False
savefig2 = False
if plotfig2 == True:
    t = sLIF_data[:, 0]
    inCur = sLIF_data[:, 1]
    oupt = sLIF_data[:, 2]
    Soupt = sLIF_data[:, 3]

    tfs = tf_data[0]
    DRM_rates = tf_data[1]
    simu_rates = tf_data[2]

    fig2width = 5.2
    fig2fgsz = (fig2width, 0.82*fig2width)
    plt.rcParams.update({'font.size': 8})

    fig = plt.figure(figsize=fig2fgsz)
    fig2gs = gridspec.GridSpec(3, 3, left=0.09, right=0.97, wspace=0., hspace=0., bottom=0.12, top=0.95, width_ratios=[6, 1.5, 6], height_ratios=[4, 1, 4])

    ax_out = plt.subplot(fig2gs[0, 0])
    ax_out.plot(t, Soupt, color='purple', linewidth=lw, clip_on=False)
    ax_out.hlines(y=np.mean(Soupt), xmin=-0.1, xmax=1.1, colors='purple', linestyle='solid', linewidth=lw-1)
    hide_axis(ax_out)
    ax_out.set_xlabel('t [s]')
    ax_out.set_ylabel(r'$\nu$ [Hz]')
    ax_out.spines['bottom'].axis.axes.tick_params(direction='in')
    ax_out.set_xlim(-0.05, 1.05)
    ax_out.set_ylim(-1., 32.)
    ax_out.set_xticks([0., 0.5, 1.0])

    ax_out1 = plt.subplot(fig2gs[0,1])
    ax_out1.axis('off')
    ax_out1.hlines(y = np.mean(Soupt), xmin=-0.1, xmax=1.1, colors='purple', linestyle='solid', linewidth=lw-1)
    ax_out1.set_ylim(-1, 32.)
    ax_out1.set_xlim(0., 1.)

    ax_main = plt.subplot(fig2gs[0,2])
    ax_main.plot(inCur, Soupt, color = color_cycle[1], linewidth=2, clip_on=False)

    ax_main.set_ylim(-1, 32.)
    ax_main.set_xlim(np.min(inCur)*1.05, np.max(inCur)*1.2)

    ax_main.hlines(y=np.mean(Soupt), xmin=np.min(inCur)*1.05-10., xmax=-270., colors=color_cycle[2], linestyle='solid', linewidth=lw-1)
    ax_main.vlines(x=np.mean(inCur), ymin=-1, ymax=0., colors='green', linestyle='solid', linewidth=lw-1)
    ax_main.axis('off')


    ax_in = plt.subplot(fig2gs[2,2])
    ax_in.plot(inCur, t, color=color_cycle[0], linewidth = 2)
    ax_in.vlines(x = np.mean(inCur), ymin=-0.05, ymax=1.1, colors='green', linestyle='solid', linewidth=lw-1)
    hide_axis(ax_in)
    ax_in.set_xlabel('Input current [pA]')
    ax_in.set_ylabel('t [s]')
    ax_in.set_ylim(-0.05, 1.05)
    ax_in.set_xlim(np.min(inCur)*1.05, np.max(inCur)*1.2)
    ax_in.set_yticks([0., 0.5, 1.])

    ax_in1 = plt.subplot(fig2gs[1,2])
    ax_in1.axis('off')
    ax_in1.vlines(x = np.mean(inCur), ymin=-0.1, ymax=1.2, colors='green', linestyle='solid', linewidth=lw-1)
    ax_in1.set_xlim(np.min(inCur)*1.05, np.max(inCur)*1.2)
    ax_in1.set_ylim(0,1)

    ax_tf = plt.subplot(fig2gs[2, 0])

    ax_tf.plot(tfs, simu_rates, linewidth=lw+3, color='darkgray', label='SIM', clip_on=True)
    ax_tf.plot(tfs, np.ones((len(tfs)))*np.mean(Soupt), color='salmon', label='SRM', linewidth=lw, clip_on=True)
    ax_tf.plot(tfs, DRM_rates, linewidth=lw, color='royalblue', label = 'DRM', clip_on=True)


    ax_tf.set_xlabel('Temporal frequency [Hz]')
    ax_tf.set_ylabel(r'$\nu$ [Hz]')
    ax_tf.spines['bottom'].axis.axes.tick_params(direction="in")
    ax_tf.legend(frameon=False, loc='upper right')
    ax_tf.set_ylim(0., 12.)
    ax_tf.set_xlim(0.3, 32)
    ax_tf.set_yticks([0., 5., 10.])
    ax_tf.set_xscale('log')
    hide_axis(ax_tf)


    fig.text(0.03, 0.96, 'A1', fontsize=fs, fontproperties=font0)
    fig.text(0.52, 0.96, 'A2', fontsize=fs, fontproperties=font0)
    fig.text(0.52, 0.5, 'A3', fontsize=fs, fontproperties=font0)
    fig.text(0.03, 0.5, 'B', fontsize=fs, fontproperties=font0)

    if savefig2 == True:
        plt.savefig('figure_2_singleLIF.png', dpi=dpi)
        plt.close()



# plot figure 3

plotfig3 = False
savefig3 = False

if plotfig3 == True:
    A0s = np.arange(16800., 23400., 100)  # shape (66, 1000, 2), A0s[36] = 20400
    A1s = np.arange(0.,9100., 100)        # shape (91, 1000, 2), A1s[40] = 4000.

    meaninCur = sLIF_A0s[36, :, 0]
    meanoupt = sLIF_A0s[36, :, 1]

    ampinCur = sLIF_A1s[40, :, 0]
    ampoupt = sLIF_A1s[40, :, 1]

    diffamps = A1s * 12.5*(20*1e-3)*1.
    diffamprates = np.mean(sLIF_A1s, axis = 1)[:, 1]

    diffmeans = (A0s-B) * 12.5*(20*1e-3)*1.
    diffmeanrates = np.mean(sLIF_A0s, axis = 1)[:, 1]

    fig3width=5.2
    fig3fgsz = (fig3width, 0.5*fig3width)
    plt.rcParams.update({'font.size': 8})

    figma = plt.figure(figsize=fig3fgsz)
    gsma = gridspec.GridSpec(2,3, left=0.13, right=0.98, wspace=0.35, hspace=0.4, bottom=0.14, top=0.94)

    ampcolor = 'purple' 
    meancolor = 'orange' 
    ax_in = plt.subplot(gsma[0,0])
    ax_in.plot(t, ampinCur, color=ampcolor, linewidth = lw)
    ax_in.hlines(y = np.mean(ampinCur), xmin=-0.05, xmax=1.1, colors=ampcolor, linestyle='solid', linewidth=lw-1)
    ax_in.plot(t, inCur, color='green', linewidth = lw)
    ax_in.hlines(y = np.mean(inCur), xmin=-0.05, xmax=1.1, colors='green', linestyle='solid', linewidth=lw-1)
    hide_axis(ax_in)
    ax_in.set_ylabel('Current [pA]')
    ax_in.set_xlim(-0.05, 1.05)
    ax_in.set_ylim(np.min(ampinCur)*1.05, np.max(ampinCur)*1.2)
    ax_in.set_xticks([0., 0.5, 1.])
    ax_in.set_yticks([-1500, -1000, -500, 0])

    ax_out = plt.subplot(gsma[0,1])
    ax_out.plot(t, ampoupt, color=ampcolor, linewidth=lw, clip_on=False)
    ax_out.hlines(y=np.mean(ampoupt), xmin=-0.1, xmax=1.1, colors=ampcolor, linestyle='solid', linewidth=lw-1)
    ax_out.plot(t, Soupt, color='green', linewidth=lw, clip_on=False)
    ax_out.hlines(y=np.mean(Soupt), xmin=-0.1, xmax=1.1, colors='green', linestyle='solid', linewidth=lw-1)
    hide_axis(ax_out)
    ax_out.set_ylabel(r'$\nu$ [Hz]')
    ax_out.spines['bottom'].axis.axes.tick_params(direction="in") 
    ax_out.set_xlim(-0.05, 1.05)
    ax_out.set_ylim(-1, 49)
    ax_out.set_xticks([0., 0.5, 1.0])

    ax_amp = plt.subplot(gsma[0,2])
    ax_amp.plot(diffamps, diffamprates, color=ampcolor, linewidth=lw)
    ax_amp.set_xlabel('Amplitude [pA]')
    ax_amp.set_ylabel(r'$\nu$ [Hz]')
    hide_axis(ax_amp)

    ax_in1 = plt.subplot(gsma[1,0])
    ax_in1.plot(t, meaninCur, color=meancolor, linewidth = lw)
    ax_in1.hlines(y = np.mean(meaninCur), xmin=-0.05, xmax=1.1, colors=meancolor, linestyle='solid', linewidth=lw-1)
    ax_in1.plot(t, inCur, color='green', linewidth = lw)
    ax_in1.hlines(y = np.mean(inCur), xmin=-0.05, xmax=1.1, colors='green', linestyle='solid', linewidth=lw-1)
    hide_axis(ax_in1)
    ax_in1.set_ylabel('Current [pA]')
    ax_in1.set_xlabel('t [s]')
    ax_in1.set_xlim(-0.05, 1.05)
    ax_in1.set_ylim(np.min(inCur)*1.05, np.max(meaninCur)*1.2)
    ax_in1.set_xticks([0., 0.5, 1.])
    ax_in1.set_yticks([-1500, -1000, -500, 0])

    ax_out1 = plt.subplot(gsma[1,1])
    ax_out1.plot(t, meanoupt, color=meancolor, linewidth=lw, clip_on=False)
    ax_out1.hlines(y=np.mean(meanoupt), xmin=-0.1, xmax=1.1, colors=meancolor, linestyle='solid', linewidth=lw-1)
    ax_out1.plot(t, Soupt, color='green', linewidth=lw, clip_on=False)
    ax_out1.hlines(y=np.mean(Soupt), xmin=-0.1, xmax=1.1, colors='green', linestyle='solid', linewidth=lw-1)
    hide_axis(ax_out1)
    ax_out1.set_xlabel('t [s]')
    ax_out1.set_ylabel(r'$\nu$ [Hz]')
    ax_out1.spines['bottom'].axis.axes.tick_params(direction="in") 
    ax_out1.set_xlim(-0.05, 1.05)
    ax_out1.set_ylim(-1, 49)
    ax_out1.set_xticks([0., 0.5, 1.0])

    ax_mean = plt.subplot(gsma[1,2])
    ax_mean.plot(diffmeans, diffmeanrates, color=meancolor, linewidth=lw)
    ax_mean.set_xlabel('Baseline [pA]')
    ax_mean.set_ylabel(r'$\nu$ [Hz]')
    hide_axis(ax_mean)

    figma.text(0.03, 0.95, 'A', fontsize=fs, fontproperties=font0)
    figma.text(0.38, 0.95, 'B', fontsize=fs, fontproperties=font0)
    figma.text(0.69, 0.95, 'C', fontsize=fs, fontproperties=font0)
    figma.text(0.03, 0.49, 'D', fontsize=fs, fontproperties=font0)
    figma.text(0.38, 0.49, 'E', fontsize=fs, fontproperties=font0)
    figma.text(0.69, 0.49, 'F', fontsize=fs, fontproperties=font0)

    if savefig3 == True:
        plt.savefig('figure_3_shift_mean_amp.png', dpi=dpi)
        plt.close()
 










