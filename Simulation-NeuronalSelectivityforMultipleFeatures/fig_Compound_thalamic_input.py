
####### Wenqing Wei ###############################################
####### Plot the compound response curve of  ######################
####### presynaptic dLGN neurons. #################################
###################################################################
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import random
import matplotlib.cm as cm
import math
from matplotlib import gridspec
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
import matplotlib.collections
import os

class Compound_thalamic_input():
    
    # The init method or constructor
    def __init__(self):
        '''Parameters:
        t: the array of time points.
        sigc: the sigma of the center Gaussian of the ON-center DoG model.
        sigs: the sigma of the surround Gaussian of the OFF-center DoG model.
        theta: the orientation of the moving grating stimulus.
        Fre_spa: the spatial frequency of the moving grating stimulus.
        Fre_tem: the temporal frequency of the moving grating stimulus.
        vf: the visual field.
        rf: the receptive field.
        K_LV: the convergence number from dLGN --> V1.
        N_dLGN: the number of dLGN cells.
        N_V1: the number of V1 cells in the recurrent network.
        N_FFI: the number of FFI sensors.
        p: the connectivity probability of the recurrent network.
        K_e: the indegree of the excitatory recurrent connection.
        K_i: the indegree of the inhibitory recurrent connection.'''
        self.t = np.linspace(0., 1./3, 500)
        self.sigc = 1.
        self.sigs = 5.
        self.theta = np.pi/3.
        self.Fre_spa = 0.08
        self.Fre_tem = 3.
        self.vf = 133.
        self.rf = 24
        self.margin = 17
        self.K_LV = 100
        self.N_dLGN = int((self.vf/self.rf)**2 * self.K_LV)
        self.N_V1 = 12500
        self.N_FFI = 1500
        self.Ne = 10000
        self.Ni = 2500
        self.p = 0.1
        self.K_e = 1000
        self.K_i = 250
        self.K_LF = 8
        self.K_FV = 320
    
    def get_random_points(self, n, vf):
        '''Get random points within the visual field.'''
        nps = np.empty((n, 2))
        for i in range(n):
            r = vf/2. * math.sqrt(random.random())
            alpha = 2 * np.pi * random.random()
            x = r * np.cos(alpha)
            y = r * np.sin(alpha)
            nps[i,0] = x
            nps[i,1] = y
        return nps
    
    def generate_locs_and_save(self):
        '''Generate random positions of the center of the Receptive fields of groups, including dLGN neurons, V1 neurons, FFI neurons.'''
        dLGN_locs = self.get_random_points(self.N_dLGN, self.vf)
        V1_locs = self.get_random_points(self.N_V1, self.vf - self.margin*2)
        FFI_locs = self.get_random_points(self.N_FFI, self.vf - self.margin*2)
        input_locs = {'dLGN_locs': dLGN_locs,
                      'V1_locs': V1_locs,
                      'FFI_locs': FFI_locs}
        np.save('input_data/input_locs.npy', np.array([input_locs]))
    

    def optimal_sf(self):
        '''calculate the optimal spatial frequency which gives the maximum value of m with (sigc, sigs) of the DoG model.'''
        sf = 2 * np.sqrt(np.log(self.sigs/self.sigc)/(self.sigs**2 - self.sigc**2))
        Fre_spa = sf/(2 * np.pi)
        return Fre_spa

    def m_max(self):
        '''calculate the maximum value of m.'''
        Fre_spa = self.optimal_sf()
        sf = 2 * np.pi * Fre_spa
        m = np.exp(-sf**2 * self.sigc**2 / 2.) - np.exp(-sf**2 * self.sigs**2 / 2.)
        return m

    def single_LGN_response_curve(self, mu, theta, sigc, sigs):
        '''The equation is to calculate the temporal response curve of a dLGN cell when the moving grating is presented.
        Parameters:
        mu: the position of the receptive field of the dLGN cell.'''
        sf = 2 * np.pi * self.Fre_spa
        m = np.exp(-sf**2 * sigc**2 / 2.) - 1 * np.exp(-sf**2 * sigs**2 / 2.)
        mmax = self.m_max()
        kv = sf * np.array([np.sin(theta), -np.cos(theta)]).reshape((2, 1))
        rt = mmax + m * np.cos(2*np.pi*self.Fre_tem*self.t - np.dot(mu, kv))
        return rt

    def compound_signal(self, nps_ON, nps_OFF, theta):
        '''The temporal compound thalamic curve is the linear summation of single response curves.
        Parameters:
        nps_ON: the positions of the receptive fields of the ON-center dLGN cells.
        nps_OFF: the positions of the receptive fields of the OFF-center dLGN cells.'''
        sum_sig = np.zeros((len(self.t)))

        for i in range(len(nps_ON)):
            rt = self.single_LGN_response_curve(nps_ON[i], theta, sigc=self.sigc, sigs=self.sigs)
            sum_sig += rt
        for j in range(len(nps_OFF)):
            rt = self.single_LGN_response_curve(nps_OFF[j], theta, sigc=self.sigs, sigs=self.sigc)
            sum_sig += rt
        return sum_sig
    
    def nearest_idx(self, nps, single_V1, nn):
        ''' get the nn nearest indices of LGN neurons to a single V1 neuron.'''
        #distance = np.linalg.norm(nps - single_V1.reshape((1,2)), axis = 1)
        c = nps - single_V1.reshape((1,2))
        distance = np.sqrt(np.einsum('ij, ij->i', c, c))
        nearest_id = distance.argsort()[:nn]
        return nearest_id

    def get_connectivity_matrix(self, nps, V1_locs, near=True, balanced=False):
        connmatrix = np.zeros((len(V1_locs), len(nps)))
        if near==False and balanced==True:
            for i in range(12500):
                onid = self.nearest_idx(nps[:int(len(nps)/2)], V1_locs[i], int(self.K_LV/2))
                offid = self.nearest_idx(nps[int(len(nps)/2):], V1_locs[i], int(self.K_LV/2)) + int(len(nps)/2)
                connmatrix[i, onid] = 1
                connmatrix[i, offid] = 1
        if near==True and balanced==False:
            for i in range(12500):
                idx = self.nearest_idx(nps, V1_locs[i], 100)
                connmatrix[i, idx] = 1
        return connmatrix

    def create_and_save_connectivities(self, save=True):
        sd = 98635
        np.random.seed(sd)

        input_locs = np.load('input_data/input_locs.npy', allow_pickle=True)[0]
        dLGN_locs = input_locs['dLGN_locs']
        V1_locs = input_locs['V1_locs']
        fullconn_LV = self.get_connectivity_matrix(dLGN_locs, V1_locs)

        conn_e = np.empty((self.N_V1, self.K_e), dtype='int')
        conn_i = np.empty((self.N_V1, self.K_i), dtype='int')
        conn_FV = np.empty((self.N_V1, self.K_FV), dtype='int')
        conn_LV = np.empty((self.N_V1, self.K_LV), dtype='int')
        
        for i in range(self.N_V1):
            idxe = np.random.choice(self.Ne, self.K_e, replace=False)
            idxi = np.random.choice(self.Ni, self.K_i, replace=False)
            conn_e[i, :] = idxe
            conn_i[i, :] = idxi + self.Ne
            
            idxfv = np.random.choice(self.N_FFI, self.K_FV, replace=False)
            conn_FV[i, :] = idxfv
            
            idxlv = np.where(fullconn_LV[i] == 1)[0]
            conn_LV[i, :] = idxlv
        if save:
            np.save('input_data/conn_e.npy', conn_e)
            np.save('input_data/conn_i.npy', conn_i)
            np.save('input_data/conn_LV.npy', conn_LV)
            np.save('input_data/conn_FV.npy', conn_FV)
            
        

class plot_grating_and_compound_input(Compound_thalamic_input):
    

    def hide_axis(self, ax):
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')

    
    def points_over_grating(self, theta):
        t = 0.1
        sf = (2. * np.pi) * self.Fre_spa
        k = sf * np.array([np.sin(theta), np.cos(theta)]).reshape((2,1))
        
        vf_range = np.arange(-self.vf/2., self.vf/2. + 0.1, 0.1)
        Xs, Ys = np.meshgrid(vf_range, vf_range)
        x = np.concatenate((Xs.reshape((-1,1)), Ys.reshape((-1,1))), axis = 1)
        F = 1 + np.cos(np.dot(x, k) - 2 * np.pi * self.Fre_tem * t)
        F = F.reshape((vf_range.size, vf_range.size))
        return vf_range, vf_range, F

    def plot_grating(self, theta, point, title = 'stimulus_grating'):
        fig = plt.figure(title)
        gs = gridspec.GridSpec(1,1)
        ax0 = plt.subplot(gs[0])
        na, nb, nFr = self.points_over_grating(theta)
        v = np.array([0., 1., 2.])
        im = plt.imshow(nFr, interpolation = 'none', cmap = cm.gray, extent = [np.amin(na), np.amax(na), np.amin(nb), np.amax(nb)], clim = (-2., 2.))
        cb = plt.colorbar(im, fraction = 0.046, pad = 0.04, ticks = v)
        cb.set_label('Light intensity (a.u.)')
        cb.ax.tick_params(labelsize = 20)
        circle = plt.Circle((0.,0.), self.vf/2., color = 'green', fill = False, linewidth = self.lw)
        plt.gcf().gca().add_artist(circle)
        plt.xlim(np.amin(na), np.amax(na))
        plt.ylim(np.amin(nb), np.amax(nb))
        plt.xlabel(r'X($^\circ$)')
        plt.ylabel(r'Y($^\circ$)')
        
    def truncate_colormap(self, cmap):
        minval=0.5
        maxval=1.0
        n=100
        new_cmap = colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
        return new_cmap

    def plot_rf_on_grating(self, gss, theta, nps_ON, nps_OFF):
        lw = 3.
        ax0 = plt.subplot(gss)
        na, nb, nFr = self.points_over_grating(theta)
        v = np.array([0., 1., 2.])
        cmap = plt.get_cmap('gray')
        new_cmap = self.truncate_colormap(cmap)
        im = plt.imshow(nFr/2., interpolation='bilinear', cmap=new_cmap, extent=[np.amin(na),np.amax(na), np.amin(nb), np.amax(nb)])
        cb = plt.colorbar(im, fraction = 0.046, pad = 0.04)
        cb.set_label('Light intensity (a.u.)')

        circle = plt.Circle((0.,0.), self.vf/2., color = 'k', fill = False, linewidth = lw)
        size = 1.84 # for center sigma=1
        patches_exc = [plt.Circle(center, size, fill = False) for center in nps_ON]
        patches_inh = [plt.Circle(center, size, fill = False) for center in nps_OFF]
        coll_exc = matplotlib.collections.PatchCollection(patches_exc, facecolor = 'brown', edgecolors = 'none', linewidth = lw)
        coll_inh = matplotlib.collections.PatchCollection(patches_inh, facecolor = 'green', edgecolors = 'none', linewidth = lw)
        coll = matplotlib.collections.PatchCollection(patches_exc + patches_inh, facecolor='none', edgecolors='k', linewidth=1)
        ax0.add_collection(coll_exc)
        ax0.add_collection(coll_inh)
        ax0.add_collection(coll)
        plt.gcf().gca().add_artist(circle)
        plt.xlim(np.amin(na), np.amax(na))
        plt.ylim(np.amin(nb), np.amax(nb))
        ax0.set_xticks([-50, 0, 50])
        ax0.set_yticks([-50, 0, 50])
        plt.xlabel(r'X [$^\circ$]')
        plt.ylabel(r'Y [$^\circ$]', labelpad=-5)

    def plot_temporal_curve(self, gs, nps_ON, nps_OFF):
        lw = 3.
        ax = plt.subplot(gs)
        color_codes = ['lightsteelblue', 'lightslategrey', 'darkcyan']
        
        degrees = np.array([0, 45, 120])
        for i in range(3):
            response = self.compound_signal(nps_ON, nps_OFF, degrees[i]*np.pi/180.)
            ax.plot(self.t, response, linewidth=lw, c=color_codes[i], label=str(degrees[i])+'$^\circ$')
        mmax = self.m_max()
        plt.hlines(y=mmax*(len(nps_ON)+len(nps_OFF)), xmin=0., xmax=0.34, linestyle='--', linewidth=lw)
        ax.set_xlim(0., 0.35)
        ax.set_xticks([0., 0.1, 0.2, 0.3])

        ax.set_yticks([65,75,85,95])
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Neuronal response')
        self.hide_axis(ax)
        leg = ax.legend(loc='lower right', frameon=False, bbox_to_anchor=(1.1, -0.05))


plot = plot_grating_and_compound_input()

if not os.path.exists('input_data'):
    os.makedirs('input_data')

# If the file of receptive field centers does not exist, generate and save.
if not os.path.exists('input_data/input_locs.npy'):
    plot.generate_locs_and_save()

# If the files of connectivity matrix do not exist, generate and save.
connfiles = ['input_data/conn_LV.npy', 
             'input_data/conn_e.npy', 
             'input_data/conn_i.npy', 
             'input_data/conn_FV.npy']
if not all(os.path.exists(f) for f in connfiles):
    print('At least one connection file is missing')
    print('Creating all connection files ...')
    plot.create_and_save_connectivities(save=True)
    print('Save all connection files - Finish!')


locs = np.load('input_data/input_locs.npy', allow_pickle=True)[0]
dLGN_locs = locs['dLGN_locs']
conn_LV = np.load('input_data/conn_LV.npy')
V1id = 2471
idx_2471 = conn_LV[V1id]
idx_on = idx_2471[idx_2471<int(plot.N_dLGN/2)]
idx_off = idx_2471[idx_2471>=int(plot.N_dLGN/2)]

# plot figure
dpi = 300
width = 5.2
height = 0.45*width
fgsz = (width, height)
plt.rcParams.update({'font.size': 8})
fig = plt.figure(figsize=fgsz)
gs = gridspec.GridSpec(1, 2, left=0.1, right=0.97, wspace=.6, bottom=0.2, top=0.9)

plot.plot_rf_on_grating(gs[0], plot.theta, dLGN_locs[idx_on[::5]], dLGN_locs[idx_off[::5]])
plot.plot_temporal_curve(gs[1], dLGN_locs[idx_on], dLGN_locs[idx_off])

fs = 10
font0 = FontProperties()
font0.set_weight('semibold')

fig.text(0.03, 0.93, 'A', fontsize=fs, fontproperties=font0)
fig.text(0.57, 0.93, 'B', fontsize=fs, fontproperties=font0)


save = False
if save == True:
    plt.savefig('figure_3_compound_dLGN.pdf')
    plt.savefig('figure_3_compound_dLGN.png', dpi=dpi)
    plt.savefig('figure_3_compound_dLGN.eps', dpi=dpi)
    plt.close()

plt.close()





