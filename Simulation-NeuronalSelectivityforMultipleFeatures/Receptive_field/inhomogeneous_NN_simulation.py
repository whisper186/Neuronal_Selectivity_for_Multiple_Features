####### Wenqing Wei ###############################################
####### 1. Create the visual stimuli                          #####
####### 2. Calculate the response of dLGN neurons to stimuli. #####
####### 3. Run the network simulation and save data.          #####
####### Be aware of that this simulation takes a lot of time  #####
####### and storage spaec.                                    #####
###################################################################

import numpy as np
import os, sys
import math
import random
import time

from importlib import reload
import inhomogeneous_NeuralNetwork; reload(inhomogeneous_NeuralNetwork)
import Response_to_visual_stimuli; reload(Response_to_visual_stimuli);
from Response_to_visual_stimuli import RF_visual_stimuli

inhNN = inhomogeneous_NeuralNetwork.inhomo_neural_Network()
Rvs = RF_visual_stimuli()

#===================================================================================
# Create the receptive field of dLGN neurons according to the position of RF center.
# Create the RF with edge and save into file to save time and storage space.
# Commend out if already exists.
#===================================================================================
locfolder = 'dLGN_RF_with_edge/'
if not os.path.exists(locfolder):
    os.makedirs(locfolder)
dLGN_locs = inhNN.dLGN_locs
dLGN_locs = np.round(dLGN_locs, 2)
# ON center cells
for i in range(0, int(len(dLGN_locs)/2)):
    srf = Rvs.create_rf_with_edge(Rvs.edge, dLGN_locs[i], Rvs.sigma)
    np.save(locfolder + 'dLGN_srf_%i.npy'%(i), srf)
# OFF center cells
for i in range(int(len(dLGN_locs)/2), len(dLGN_locs)):
    srf = Rvs.create_rf_with_edge(Rvs.edge, dLGN_locs[i], Rvs.sigma[::-1])
    np.save(locfolder + 'dLGN_srf_%i.npy'%(i), srf)

#===================================================================================
# Create square stimuli and calculate the responses of dLGN neurons
# The squares were appeard in one of the locations covering the entire visual field
# Commend out if already exists.
#===================================================================================
Nstimu = int(Rvs.Ngrid**2)*2
savefolder = 'square_stimuli_Nframe_%i/'%(Nstimu)
if not os.path.exists(savefolder):
    os.makedirs(savefolder)

sqrstiFolder = 'square_stimulus_array/'
if not os.path.exists(savefolder + sqrstiFolder):
    os.makedirs(savefolder + sqrstiFolder)

# create the array of the position and light information of square stimuli, every possible location with light/dark spot appeared once.
ind = np.arange(Rvs.Ngrid, dtype='int')
indx, indy = np.meshgrid(ind, ind)
indxy = np.zeros((int(Rvs.Ngrid**2), 2), dtype='int')
indxy[:, 0] = indy.reshape((1, -1))
indxy[:, 1] = indx.reshape((1, -1))
xylight = np.zeros((int(2*Rvs.Ngrid**2), 3), dtype='int')
xylight[: int(Rvs.Ngrid**2), :2] = indxy
xylight[int(Rvs.Ngrid**2) :, :2] = indxy
xylight[: int(Rvs.Ngrid**2), 2] = 0  # 0 for dark square
xylight[int(Rvs.Ngrid**2) :, 2] = 1  # 1 for light square

np.random.shuffle(xylight)
np.save(savefolder + 'randint_indxidy_light_%i.npy'%(Nstimu), xylight)

# create square stimuli according to the previous information and calculate the response of dLGN neurons
randxylight = np.load(savefolder + 'randint_indxidy_light_%i.npy'%(Nstimu))
dLGNresponse = np.empty((Nstimu, inhNN.all_n))
for si in range(Nstimu):
    evf = Rvs.create_square_stimulus(randxylight[si, 0], randxylight[si, 1], light=randxylight[si, 2])
    np.save(savefolder + sqrstiFolder + 'square_stimulus_%i.npy'%i, evf)
    response = []
    for i in range(inhNN.all_n):
        srf = np.load(locfolder + 'dLGN_srf_%i.npy'%(i))
        erf = Rvs.load_and_fullfill_gaussian_from_previous(srf, inhNN.dLGN_locs[i])
        resp = Rvs.response_to_visual_stimulus(evf, erf)
        response.append(resp)
    dLGNresponse[si, :] = response

np.save(savefolder + 'dLGN_response_%i.npy'%(Nstimu), dLGNresponse)


#===========================================================
# Run network simulation for square stimulus
#===========================================================
frame_nr = int(Rvs.Ngrid**2)*2
time_bin = inhNN.time_bin  # ms, each square stimulus is presented for time_bin ms
recordT = time_bin * frame_nr
rate_times = np.arange(inhNN.t_drop, recordT + inhNN.t_drop, time_bin)
rate_values = np.load(savefolder + 'dLGN_response_%i.npy'%(frame_nr))
dLGN_rate_values = inhNN.A * (2 * rate_values + inhNN.mmax)

all_times = np.append(np.array([inhNN.dt]), rate_times)
all_dLGN_rates = np.zeros((frame_nr + 1, inhNN.all_n))
all_dLGN_rates[0, :] = inhNN.A * inhNN.mmax
all_dLGN_rates[1:, :] = dLGN_rate_values

dSD, dSD_dLGN, dSD_FFI = inhNN.inhomo_network_simulation(all_times, all_dLGN_rates, inhNN.ext_rate)
np.save(savefolder + 'squareStimu_result_dLGN_tbin%ims_2r.npy'%(time_bin), np.array([dSD_dLGN['times'], dSD_dLGN['senders']]).T)
np.save(savefolder + 'squareStimu_result_V1_tbin%ims_2r.npy'%(time_bin), np.array([dSD['times'], dSD['senders']]).T)




#############################################################################
# Spot stimuli   ############################################################
#############################################################################
# create spot stimuli and calculate the response of dLGN cells
Nframe_spot = 20000
spotfolder = 'sparseNoise_stimuli_Nframe_%i/'%(Nframe_spot)
if not os.path.exists(spotfolder):
    os.makedirs(spotfolder)

spotstiFolder = 'sparseNoise_stimulus_array/'
if not os.path.exists(spotfolder + spotstiFolder):
    os.makedirs(spotfolder + spotstiFolder)

radi = 1.84
nspot = 500    # around 0.2 * (Rvs.vf+Rvs.edge*2)**2/(np.pi*radi**2)
dLGN_response = np.empty((Nframe_spot, len(dLGN_locs)))
background = np.zeros((Rvs.lx, Rvs.lx))
for i in range(Nframe_spot):
    evf = Rvs.create_spots_stimulus(nlspot=int(nspot/2), ndspot=int(nspot/2), radi=radi)
    np.save(spotfolder + spotstiFolder + 'sparseNoise_stimulus_%i.npy'%i, evf)
    background += evf
    response = []
    for idx in range(len(dLGN_locs)):
        srf = np.load(locfolder + 'dLGN_srf_%i.npy'%(idx))
        erf = Rvs.load_and_fullfill_gaussian_from_previous(srf, inhNN.dLGN_locs[idx])
        resp = Rvs.response_to_visual_stimulus(evf, erf)
        response.append(resp)
    dLGN_response[i, :] = response
background = background/Nframe_spot
np.save(spotfolder + 'dLGN_response_%i.npy'%(Nframe_spot), dLGN_response)
np.save(spotfolder + 'sparseNoise_stimulus_background.npy', background)
#===========================================================
# Run network simulation for sparse noise stimuli
#===========================================================
time_bin = 33.  # ms, each stimulus is presented for time_bin ms
recordT = time_bin * Nframe_spot
rate_times = np.arange(inhNN.t_drop, recordT + inhNN.t_drop, time_bin)
rate_values = np.load(spotfolder + 'dLGN_response_%i.npy'%(Nframe_spot))
dLGN_rate_values = inhNN.A * (2 * rate_values + inhNN.mmax)

all_times = np.append(np.array([inhNN.dt]), rate_times)
all_dLGN_rates = np.zeros((frame_nr + 1, inhNN.all_n))
all_dLGN_rates[0, :] = inhNN.A * inhNN.mmax
all_dLGN_rates[1:, :] = dLGN_rate_values

dSD, dSD_dLGN, dSD_FFI = inhNN.inhomo_network_simulation(all_times, all_dLGN_rates, inhNN.ext_rate)
np.save(spotfolder + 'sparseNoiseStimu_result_dLGN_tbin%ims_2r.npy'%(time_bin), np.array([dSD_dLGN['times'], dSD_dLGN['senders']]).T)
np.save(spotfolder + 'sparseNoiseStimu_result_V1_tbin%ims_2r.npy'%(time_bin), np.array([dSD['times'], dSD['senders']]).T)

print ('V1 mean rates: %.3f Hz'%(len(dSD['times'])/(recordT*len(inhNN.V1_locs))))
print ('dLGN mean rates: %.3f Hz'%(len(dSD_dLGN['times'])/(recordT * len(inhNN.dLGN_locs))))

