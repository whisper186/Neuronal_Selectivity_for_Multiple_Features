#### Wenqing Wei


import numpy as np
import time

def timeConversion(t):
    t = int(np.round(t))
    # seconds
    if t < 60:
        return '%2is'%t
    else:
        secs = t%60
        t = (t - secs)/60
        strng = '%2is'%secs
        #minutes
        if t < 60:
            return '%2im:'%t + strng
        else:
            mins = t%60
            t = (t - mins)/60
            strng = '%2im:'%mins + strng
            #hours
            if t < 24:
                return '%2ih:'%t + strng
            else:
                hrs = t%24
                t = (t - hrs)/24
                return '%2id:'%t + '%2ih:'%hrs + strng

def hide_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def OSI_PO(rates, angles):

    OSIs, POs, OSVs = [], [], []
    for i in range(len(rates)):
        OSV = np.sum(rates[i] * np.exp(2 * np.pi * 1j * angles/180.))
        nOSV = np.sum(rates[i])
        
        if nOSV == 0:
            OSV, nOSV = 0, 1
        
        osvs = OSV/nOSV
        PO = np.angle(osvs)
        if PO < 0:
            PO = PO + 2 * np.pi
        PO = PO/(2 * np.pi) * 180.
        OSI = abs(osvs)
        POs.append(PO)
        OSIs.append(OSI)
        OSVs.append(osvs)
    
    return np.array(POs), np.array(OSIs), np.array(OSVs)
        
def calcOS(list_rates, angles):
    
    list_mOSIs, list_POs, list_OSIs, list_OSVs = [], [], [], []
    
    for idx in range(len(list_rates)):
        POs, OSIs, OSVs = OSI_PO(list_rates[idx], angles)
        
        list_mOSIs.append(np.mean(OSIs))
        list_POs.append(POs)
        list_OSIs.append(OSIs)
        list_OSVs.append(OSVs)
    
    return list_mOSIs, list_POs, list_OSIs, list_OSVs
    

color_cycle = ['red', 'blue', 'green', 'salmon', 'lightblue', 'mediumslateblue', 'darkcyan']
