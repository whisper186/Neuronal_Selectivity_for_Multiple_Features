######## Parameters with network simulation ##############

import numpy as np
import time
import random

# General simulation parameters

n_vp=20   # total number of virtual processes
dt = 0.1    # simulation resolution (ms)

# External input rate
ext_rate = 8350.  # the firing rate of external input (Hz)
J_ext = 0.1    # external synaptic weight


# Visual field, stimulus parameters 
vf = 133.   # diameter of the large visual field (degree or °)
rf = 24.    # diameter of the receptive field of V1 neurons (degree or °)
rSafetyMargin = 17. # the difference of the radius between the large visual field and the visual field for V1 cells to minimize the border effect (degree or °)
mu = np.array([0., 0.]) # the center point of the visual field.

Fre_spa = 0.08     # spatial frequency (cycles per degree, i.e. cpd)
Fre_tem = 3.       # temporal frequency (Hz)

# Number of neurons
n = 100     # number of dLGN sensors that converge to a single V1 neuron
all_n = int((vf/rf)**2 * n)   # the total number of dLGN sensors on the whole visual field
N_FFI = 1500   # total number of FFI sensors in the network

# Parameters of the leaky integrate and fire neuron
tau_m = 20.    # membrane time constant (ms)
t_ref = 2.     # refractory period (ms)
delay = 1.5    # synaptic delay (ms)
E_L = 0.0      # resting membrane potential (mV)
V_reset = 0.0  # reset potential of the membrane (mV)
V_th = 20.     # spike threshold (mV)
C_m = 250.     # membrane capacitance (pF)


# Parameters for cortical recurrent network, Brunel's network

N = 12500      # total number of V1 neurons in the network
fe = 0.8       # excitatory fraction
Ne = int(N*fe) # excitatory neuron population size
Ni = int(N-Ne) # inhibitory neuron population size
J_E = 0.2      # recurrent excitatory connection weight
g = -8         # excitation-inhibition ratio
J_I = J_E * g  # recurrent inhibitory connection weight

p = 0.1        # connection probability
ind_E = int(p*Ne)   # excitatory inputs
ind_I = int(p*Ni)   # inhibitory inputs

# Parameters for network simulation

n_LV = n       # the number of dLGN sensors that converge to a single V1 neuron
n_LF = 8       # the number of dLGN sensors that converge to a single FFI neuron
n_FFI_V1 = 320 # the number of FFI sensors that converge to a single V1 neuron

J_LV = 2.0     # the synaptic weight of dLGN --> V1 (mV)
J_LF = 2.0     # the synaptic weight of dLGN --> FFI (mV)
J_FFI_V1 = -1.6# the synaptic weight of FFI --> V1 (mV)

A = 100.       # the firing rate of dLGN sensors in response to the mean luminosity of the stimulus s0 (Hz)


t_drop = 200.  # Simulation time, discard the first 200 ms
T = 6*1000.    # Simulation time, recording time (ms)

deg_range = np.arange(0., 180., 15)


def mcontrast(Fre_spa, sigv, c):
    sf = 2 * np.pi * Fre_spa
    return np.exp(-sf ** 2 * sigv[0]**2/2.) - c * np.exp(-sf**2 * sigv[1] **2/2.)


def optimal_sf(sigv):
    sf = 2 * np.sqrt(np.log(sigv[1]/sigv[0])/(sigv[1]**2 - sigv[0]**2))
    Fre_spa = sf/(2 * np.pi)
    return Fre_spa

def m_max(sigv, c):
    Fre_spa = optimal_sf(sigv)
    m = mcontrast(Fre_spa, sigv, c)
    return m

mmax = m_max(np.array([1., 5.]), 1.)

