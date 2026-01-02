####### Wenqing Wei ###############################################
####### Import the dynamic rate model module ######################
####### Run the analytical simulation for a full harmonic #########
####### circle (60 evenly distributed time steps) across ##########
####### all stimulus orientations, save data to files. ############
###################################################################

import os, sys
import numpy as np

from multiprocessing import Queue, Process

from importlib import reload
import rate_model; reload(rate_model)

obj = rate_model.single_neuron_mf()

print ('=== loading connectivity ===')
obj.loadConn()
obj.loadConnFFIV1()

print ('=== run rate model ===')

# First load amplitudes and phases of compound signals.
path = '../'
com_folder = 'output_data/simu_spatialFrequencies/sFs_amps_phases/'
amps = np.load(path + com_folder + 'amps_0.08_1-5_1.npy')
phases = np.load(path + com_folder + 'phases_0.08_1-5_1.npy')
FFI_amps = np.load(path + com_folder + 'FFI_amps_0.08_1-5_1.npy')
FFI_phases = np.load(path + com_folder + 'FFI_phases_0.08_1-5_1.npy')

angles = np.arange(0., 180., 15)
nt = 60

# the attenuated parameter is determined by the temporal frequency
attFactor = 1/np.sqrt(1+(2*np.pi*obj.tf*obj.tau_m)**2)
print ('tau_m: %.2f; attFactor: %.4f'%(obj.tau_m, attFactor))

print ('t_max = ', 1./obj.tf)

queue = Queue()

def doAngle(a):
    
    print ('temporal frequency -- %i'%obj.tf)
    v = np.zeros((obj.N, nt))
    Nu = np.zeros((obj.N, nt))
    v_FFI = np.zeros((obj.N_FFI, nt))
    
    init = 'zero'
    
    if a == 0:
        prnt = True
    else:
        prnt = False
    
    for i in np.arange(nt):
        t = 1./obj.tf * float(i)/nt   
        
        # get the input current to V1 neurons
        Nu[:, i] = obj.A * (obj.m * obj.n_lv + amps[:,a] * np.cos(t * obj.tf * 2*np.pi - phases[:, a]))
        # get the input current to FFI neurons
        Nu_FFI = obj.A * (obj.m * obj.n_lf + FFI_amps[:,a] * np.cos(t * obj.tf * 2*np.pi - FFI_phases[:,a]))
        
        # calculate the rate of FFI neurons
        v_FFI[:, i] = obj.FFI_rate(Nu_FFI)
        
        if a == 0:
            print ('\n *', i, '*| t = ', t)
        
        # get the rate of V1 neurons
        vr = obj.siegPredict(Nu[:,i], v_FFI[:,i], plot=False, der_tol=1e-6, s_step=10, init=init, prnt=prnt)
        
        v[:,i] = vr
        
        init = v[:,i]
    
    queue.put((a, v, Nu, v_FFI))

A = angles.size
v_all = np.zeros((A, obj.N, nt))  # rates of V1 neurons
Nu_all = np.zeros((A, obj.N, nt))  # input rates of dLGN neurons
v_FFI_all = np.zeros((A, obj.N_FFI, nt))  # input rates of FFI neurons

max_proc = 12
a, n = 0, 0
while n < A:
    if (a-n) == max_proc or a == A:
        
        ap, v, Nu, v_FFI = queue.get()
        v_all[ap, :, :] = v
        Nu_all[ap, :, :] = Nu
        v_FFI_all[ap, :, :] = v_FFI
        
        n += 1
        
        print ('finished %i/%i'%(n, A))
    
    if a < A:
        p = Process(target=doAngle,
                    args=(a,))
        p.start()
        
        a += 1

save_path = 'output_data/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(save_path + 'full_rates_nt60.npy', v_all)
np.save(save_path + 'lgn_input_nt60.npy', Nu_all)
np.save(save_path + 'FFI_rates_nt60.npy', v_FFI_all)
        


