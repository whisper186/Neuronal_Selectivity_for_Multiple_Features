# calculate the circular correlation 

import numpy as np




def get_differences(POs_1, POs_2):
    POs_1, POs_2 = POs_1 * np.pi/180., POs_2 * np.pi/180. # convert degree to pi 
    diff_1, diff_2 = [], []
    for i in range(len(POs_1) - 1):
        diff_1.append(POs_1[i] - POs_1[i+1::])
        diff_2.append(POs_2[i] - POs_2[i+1::])
    diff_1 = np.concatenate(diff_1, axis = 0)
    diff_2 = np.concatenate(diff_2, axis = 0)
    return diff_1, diff_2

def circular_correlation(POs_1, POs_2):
    diff_1, diff_2 = get_differences(POs_1, POs_2)
    
    cc_up = np.sum(np.sin(diff_1) * np.sin(diff_2))
    cc_bottom = np.sqrt(np.sum(np.sin(diff_1)**2) * np.sum(np.sin(diff_2)**2))
    
    return cc_up/cc_bottom


def plot_without_top_right_axis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

def plot_PO_corr(ax, x_PO, y_PO, color=None, xlabel=None, ylabel=None):
    corr = np.corrcoef(x_PO, y_PO)
    #plt.title('CC = %.2f'%corr[0,1])
    ax.plot(x_PO, y_PO, '.', color=color)
    plot_without_top_right_axis(ax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(0., 180.)
    plt.ylim(0., 180.)
    plt.tight_layout()


def nps_input_inside_rf(nps, rf, single_V1, nn):
    c = nps - single_V1.reshape((1, 2))
    distance = np.sqrt(np.einsum('ij, ij->i', c, c))
    inside_ids = np.where(distance <=rf/2.)[0]
    randomid = random.sample(range(0, len(inside_ids)), nn)
    return inside_ids[randomid]


def calculate_circular_difference(POs_1, POs_2):
    sindiff = np.sin((POs_1 - POs_2) * np.pi/180.)
    anglediff = np.arcsin(sindiff)
    anglediff[anglediff<0] = -anglediff[anglediff<0]
    degdiff = anglediff/np.pi*180.
    return degdiff

