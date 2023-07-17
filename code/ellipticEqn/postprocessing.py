"""
Post-processing of the FEM results.
Plot, data saving and the error calculations.
"""
import numpy as np
import sys
from mesh import TriMesh2d, BoundaryData2d
import fem2d
import scipy.sparse.linalg
import time
from numba import jit
np.set_printoptions(threshold=sys.maxsize)

import matplotlib.pyplot as plt
import plot


def convert_2d_data(x1d_FE, y1d_FE, Pb, vecx):
    """
    convert the FEM data to normal 2d data
    This only works for rectangular mesh grids.
    """
    x2d_FE, y2d_FE = np.meshgrid(x1d_FE, y1d_FE, indexing='ij')
    val2d_FE = np.zeros(x2d_FE.shape)

    for i in range(Pb.shape[1]):
        x = Pb[0,i]
        y = Pb[1,i]
        val = vecx[i]
        idx_x = np.argmin(np.abs(x1d_FE - x))
        idx_y = np.argmin(np.abs(y1d_FE - y))
        val2d_FE[idx_x, idx_y] = val
    return x2d_FE, y2d_FE, val2d_FE

def plot_2d_data(x2d, y2d, val2d):
    fig = plt.figure(num=1)
    plotfig = plot.PlotFig()
    plotfig.add_conf_ax(fig, 1, 1, 1,\
                        x2d, y2d, val2d, \
                        title="2D FEM data contour", \
                        xlabel=r'$x(\rm{m})$', 
                        ylabel=r'$y(\rm{m})$', 
                        lines=30)
    plt.show()
