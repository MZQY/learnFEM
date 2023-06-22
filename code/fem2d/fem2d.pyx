"""
FEM basis functions and Gaussian quadrature
ymma98
"""

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt
from libc.stdio cimport printf
from libc.stdlib cimport exit

# import sys, os
# sys.path.append(os.path.dirname(os.getcwd()) + '/basis/')
# print(sys.path)
cimport basis.basis


# def get_matA(Nlb,  N,  Nb,  matP,  matT, \
#          matTb,  matPb, \
#          trial_func_num,  test_func_num,\
#          trial_func_ndx,   trial_func_ndy,  \
#          test_func_ndx,   test_func_ndy, matA):
#     _get_matA(coeff_function, Nlb,  N,  Nb,  matP,  matT, \
#          matTb,  matPb, \
#          trial_func_num,  test_func_num,\
#          trial_func_ndx,   trial_func_ndy,  \
#          test_func_ndx,   test_func_ndy, matA)

cdef double coeff_function(double x, double y):
    return 2.


# https://stackoverflow.com/questions/18348083/passing-cython-function-to-cython-function
ctypedef double (*COEFF_FUNC)(double x, double y)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _get_matA (COEFF_FUNC cfunc, \
        int Nlb, int N, int Nb,\
        double[:,:] matP, double[:,:] CmatT, \
        double[:,:] CmatTb, double[:,:] matPb, \
        int trial_func_num, int test_func_num,\
        int trial_func_ndx,  int trial_func_ndy,  \
        int test_func_ndx,  int test_func_ndy, double[:,:] matA) nogil:
    cdef int i =0;
    cdef int j;
    for i in range(0,100):
        j=i;

    
    pass





