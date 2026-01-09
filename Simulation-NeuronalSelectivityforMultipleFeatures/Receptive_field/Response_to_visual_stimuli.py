####### Wenqing Wei ###############################################
####### 1. Create the stimuli of sparse noise.  ###################
####### 2. Draw the receptive fields of dLGN neurons in pixels. ###
####### 3. Calculate responses of dLGN neurons to stimuli. ########
###################################################################

import numpy as np
import sys
import math
import random
import time
import os


class RF_visual_stimuli():

    # The init method or constructor
    def __init__(self):
        '''Parameters:
        vf: the size of the visual field. It is an even number to avoid non-integer number of vf/2.
        delta: the size of each pixel.
        x: the axis value of each pixel in the visual field.
        sigma: the sigmas of center and surround Gaussion of the DoG model of dLGN neurons.
        edge: the rough edge of the surround Gaussian. Normally considered as 3*sigma for Gaussian function, here use the factor of 4 for safety reason.
        square: the width of square stimuli as described in Lien&Scanziani, 2013
        '''
        self.vf = 134.
        self.delta = 0.2
        self.x = np.arange(-self.vf/2, self.vf/2+self.delta, self.delta)
        self.x = np.round(self.x)
        self.sigma = np.array([1., 5.])
        self.edge = 4*self.sigma[1]
        self.lx = len(self.x) + int(self.edge/self.delta)*2 # avoid the bias at the edge of the visual field
        #self.radi = 1.84
        self.square = 5.
        self.grid = self.square/self.delta
        self.Ngrid = round((self.vf + 2*self.edge)/self.square) # the number of squares
    
    def get_random_points(self, n, vfd):
        '''Get random points within the visual field.
        Parameters:
        n: the number of points.
        vfd: the size of visual field.
        '''
        nps = np.empty((n, 2))
        for i in range(n):
            r = vfd/2. * math.sqrt(random.random())
            alpha = 2 * np.pi * random.random()
            ix = r * np.cos(alpha)
            iy = r * np.sin(alpha)
            nps[i, 0] = ix
            nps[i, 1] = iy
        return nps

    def create_spots_stimulus(self, nlspot, ndspot, radi=1.84):
        '''Create sparse noise stimulus.
        Parameters:
        radi: the radius of individual spot on the background.
        nlspot: the number of light spots.
        ndspot: the number of dark spots.
        '''
        # generate the stimulus with grey background
        evf = np.ones((self.lx, self.lx)) * 0.5
        locx = np.arange(-self.vf/2 - self.edge, self.vf/2 + self.edge + self.delta, self.delta)
        locy = locx[::-1]

        #======================================
        # light spots
        #======================================
        # first get the center position of light spots
        llocs = self.get_random_points(nlspot, vfd = self.vf+self.edge*2)
        # for each spot, first get the i, js of the center point, 
        # if the distance of the pixel to the center point is smaller than radius, set it to 1, i.e. grey + 0.5
        for idx in range(len(llocs)):
            mu = llocs[idx]
            i = int(round((-mu[1] + self.vf/2.+self.edge)/self.delta))
            j = int(round((mu[0] + self.vf/2.+self.edge)/self.delta))
            cr = round(radi)
            for ii in range(i - int(cr/self.delta), i+int(cr/self.delta)+1):
                if ii >= 0 and ii < len(locx):
                    for jj in range(j - int(cr/self.delta), j + int(cr/self.delta)+1):
                        if jj >= 0 and jj < len(locy):
                            dist = np.sqrt((locy[ii]-mu[1])**2 + (locx[jj]-mu[0])**2)
                            if dist <= radi:
                                evf[ii, jj] += 0.5
        
        #======================================
        # dark spots
        #======================================
        dlocs = self.get_random_points(ndspot, vfd = self.vf+self.edge*2)
        # for each spot, first get the i, js of the center point, 
        # if the distance of the pixel to the center point is smaller than radius, set it to 0, i.e. grey - 0.5
        for idx in range(len(dlocs)):
            mu = dlocs[idx]
            i = int(round((-mu[1] + self.vf/2.+self.edge)/self.delta)+1)
            j = int(round((mu[0] + self.vf/2.+self.edge)/self.delta))
            cr = round(radi)
            for ii in range(i - int(cr/self.delta), i + int(cr/self.delta)+1):
                if ii >= 0 and ii < len(locx):
                    for jj in range(j - int(cr/self.delta), j + int(cr/self.delta)+1):
                        if jj >= 0 and jj < len(locy):
                            dist = np.sqrt((locy[ii]-mu[1])**2 + (locx[jj]-mu[0])**2)
                            if dist <= radi:
                                evf[ii, jj] += -0.5
        
        evf[evf<0] = 0
        evf[evf>1] = 1

        return evf

    def create_square_stimulus(self, idx, idy, light=True):
        '''Create square stimulus.
        The stimuli used to map receptive fields consisted of individually presented black (minimum luminance) or white squares (full luminance) against a gray background of mean luminance. 
        Note that the squares are not overlapping with each other. If you do not want to set this condition, simply change the code to:
        evf[idx : idx+grid, idy : idy+grid] = 0. or 1. (idx and idy in the range between (0, int(lx-grid)))
        In this case, you are creating the square with size (square, square) at a random position
        Parameters:
        idx: the index of the start point on the column. It is in the range between (0, Ngrid)
        idy: the index of the start point on the row.
        light: boolean, True for light square, False for dark square.
        '''
        # First create the grey background.
        evf = np.ones((self.lx, self.lx)) * 0.5

        if light == True:
            evf[int(idx*self.grid):int((idx+1)*self.grid), int(idy*self.grid) : int((idy+1)*self.grid)] = 1.
        elif light == False:
            evf[int(idx*self.grid):int((idx+1)*self.grid), int(idy*self.grid) : int((idy+1)*self.grid)] = 0.
        else:
            raise UserWarning('light variable should be boolean, either True or False')
        
        return evf

    def create_locs_of_square_covers_entire_vf(self, save=False):
        '''Create the location indices of light and dark squares used for visual stimuli. The squares were 5 degrees in width and appeared in one of 35*35 locations covering the entire visual field.
        Parameters:
        save: boolean, whether save the random shuffled indices or not.
        Return array shape (Ngrid**2 * 2, 3), [row indices, column indices, 0(dark) or 1(light)]
        Each combination in this file can be used to create/recover a square stimulus.
        '''
        ind = np.arange(self.Ngrid, dtype='int')
        indx, indy = np.meshgrid(ind, ind)
        indxy = np.zeros((int(self.Ngrid**2), 2), dtype='int')
        indxy[:, 0] = indy.reshape((1, -1))
        indxy[:, 1] = indx.reshape((1, -1))
        xylight = np.zeros((int(2*self.Ngrid**2), 3), dtype='int')
        xylight[:int(self.Ngrid**2), :2] = indxy
        xylight[int(self.Ngrid**2):, :2] = indxy
        xylight[:int(self.Ngrid**2), 2] = 0
        xylight[int(self.Ngrid**2):, 2] = 1

        # randomize the square stimuli
        np.random.shuffle(xylight)
        savefolder = 'randint_randlight_%i'%(len(xylight))
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        
        if save == True:
            np.save(savefolder + 'square_stimuli_randint_randlight_%i.npy'%(len(xylight)), xylight)
        return xylight

    
    def create_multiple_square_on_background(idarray, light=True):
        '''Create square stimulus.
        There are more than one square on the background in the visual field. All squares are of the same type.
        Parameters:
        idarray: the array of the start points of the squares.
        '''
        # First create the grey background
        evf = np.ones((self.lx, self.lx)) * 0.5
        if light == True:
            for ids in range(len(idarray)):
                evf[int(idarray[ids, 0]*self.grid) : int((idarray[ids, 0]+1)*self.grid), int(idarray[ids, 1]*self.grid) : int((idarray[ids, 1]+1)*self.grid)] = 1.
        elif light == False:
            for ids in range(len(idarray)):
                evf[int(idarray[ids, 0]*self.grid) : int((idarray[ids, 0]+1)*self.grid), int(idarray[ids, 1]*self.grid) : int((idarray[ids, 1]+1)*self.grid)] = 0.
        else:
            raise UserWarning('light variable should be boolean, either True or False')
        return evf

    
    def receptive_field_2d(self, xv, yv, mu, sigv):
        '''Create two dimensional receptive field of dLGN neurons with DoG model.
        Parameters:
        xv, yv: the 2d coordinate on the visual field.
        mu: the position of the center of the neuron's RF.
        sigv: the sigmas of the ON and OFF Gaussian subfields.'''
        rf_pl = 1./(2 * np.pi * sigv[0]**2) * np.exp(-1./(2 * sigv[0]**2) * ((xv-mu[0])**2 + (yv - mu[1])**2))
        rf_mi = 1./(2 * np.pi * sigv[1]**2) * np.exp(-1./(2 * sigv[1]**2) * ((xv-mu[0])**2 + (yv - mu[1])**2))
        rf = rf_pl - rf_mi
        return rf
    
    def create_rf_on_entire_vf(self, mu, sigv):
        '''Create two dimensional RF of dLGN neurons covers the entire visual field.
        Parameters:
        mu: the position of the center of the neuron's RF.
        sigv: the sigmas of the ON and OFF Gaussian subfields.'''
        xs = np.arange(-self.vf/2-self.edge, self.vf/2+self.edge+self.delta, self.delta)
        ys = np.arange(-self.vf/2-self.edge, self.vf/2+self.edge+self.delta, self.delta)
        xv, yv = np.meshgrid(xs, ys)
        erf = self.receptive_field_2d(xv, yv[::-1], mu, sigv)
        return erf

    def create_rf_with_edge(self, edge, mu, sigv):
        '''Create two dimensional RF of dLGN neurons within the area of edge.
        Parameters:
        edge: the extension of the RF from the center.
        mu: the position of the center of the neuron's RF.
        sigv: the sigmas of the ON and OFF Gaussian subfields.'''
        #ne = int(edge/self.delta)
        xs = np.arange(mu[0]-edge, mu[0]+edge, self.delta)
        ys = np.arange(mu[0]-edge, mu[1]+edge, self.delta)
        xv, yv = np.meshgrid(xs, ys)
        edge_rf = self.receptive_field_2d(xv, yv[::-1], mu, sigv)
        return edge_rf
    

    def load_and_fullfill_gaussian_from_previous(self, edge_rf, mu):
        '''Put the receptive field with edge that generated from the previous function in the entire visual field.
        Parameters:
        edge_rf: the receptive field of the dLGN neuron with edge.
        mu: the center position of the neuron's RF.'''
        edge = int(edge_rf.shape[0]*self.delta/2)
        ne = int(edge/self.delta)
        erf = np.zeros((self.lx, self.lx))
        i = int(round((-mu[1] + self.vf/2+edge)/self.delta)+1)
        j = int(round((mu[0] + self.vf/2.+edge)/self.delta))
        erf[i-ne:i+ne, j-ne:j+ne] = edge_rf
        return erf
    
    def response_to_visual_stimulus(self, stimulus, rf):
        '''Calculate the response of a Neuron to the visual stimulus.
        Parameters:
        stimulus: the visual stimulus
        rf: the receptive field.
        stimulus and rf should have the same shape.'''
        if stimulus.shape == rf.shape:
            respConv = np.multiply(stimulus , rf)
            response = np.sum(respConv)*self.delta**2
        else:
            raise UserWarning('The shape of stimulus and rf should be the same!')
        return response


class load_locs_and_connections():
    def __init__(self):
        pass
    def load_locations(self, lockey=None):
        '''Load the locations of neurons that saved from previous simulations.
        Parameters:
        lockey: the key of the location file. ['dLGN', 'V1', 'FFI']
        The keys of the loaded file are ['dLGN_locs', 'V1_locs', 'FFI_locs'].
        If lockey=None, return the loaded dictionary which contains all locations of all neurons.
        Otherwise, return the locations of the given neuron type.'''
        locations = np.load('../input_data/input_locs.npy', allow_pickle=True)[0]
        if lockey == None:
            return locations
        elif lockey == 'dLGN':
            return locations['dLGN_locs']
        elif lockey == 'V1':
            return locations['V1_locs']
        elif lockey == 'FFI':
            return locations['FFI_locs']
        else:
            raise UserWarning('Wrong neuron type, please double check!')
    
    def load_connections(self, connkey=None):
        '''Load the connections between neuron groups that saved from previous simulations.
        Parameters:
        connkey: tell the function which to load.
        '''
        connpath = '../input_data/'
        if connkey == 'rec_e':
            connections = np.load(connpath + 'conn_e.npy')
            print ('load excitatory recurrent connections, shape '+str(connections.shape))
        elif connkey == 'rec_i':
            connections = np.load(connpath + 'conn_i.npy')
            print ('load inhibitory recurrent connections, shape '+str(connections.shape))
        elif connkey == 'dLGNtoV1':
            connections = np.load(connpath + 'conn_LV.npy')
            print ('load the connections from dLGN to V1 neurons, shape '+str(connections.shape))
        elif connkey == 'FFItoV1':
            connections = np.load(connpath + 'conn_FV.npy')
            print ('load the connections from FFI to V1 neurons, shape '+str(connections.shape))
        elif connkey == 'all':
            conne = np.load(connpath + 'conn_e.npy')
            conni = np.load(connpath + 'conn_i.npy')
            connLV = np.load(connpath + 'conn_LV.npy')
            connFV = np.load(connpath + 'conn_FV.npy')
            connections = {'rec_e': conne, 
                           'rec_i': conni,
                           'dLGNtoV1': connLV,
                           'FFItoV1': connFV}
            print ('load all connections, put in dictionary.')
        else:
            raise UserWarning('Wrong condition, the condition should be one of [\'rec_e\', \'rec_i\', \'dLGNtoV1\', \'FFItoV1\', \'all\']')
        
        return connections







