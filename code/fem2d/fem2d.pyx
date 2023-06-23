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
cimport basis.basis as basis


cdef double coeff_function(double x, double y):
    return 2.


# https://stackoverflow.com/questions/18348083/passing-cython-function-to-cython-function
ctypedef double (*COEFF_FUNC)(double x, double y)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _get_matA (COEFF_FUNC cfunc, \
        int N, int Nb,\
        double[:,:] matP, int[:,:] CmatT, \
        int[:,:] CmatTb_trial, \
        int[:,:] CmatTb_test,  \
        int Nlb_trial,  int Nlb_test, \
        int trial_func_ndx,  int trial_func_ndy,  \
        int test_func_ndx,  int test_func_ndy, \
        double[:,:] matA) nogil:
    cdef int n=0
    for n in prange(N, nogil=True):
        _matA_element_loop(cfunc, \
        n, Nb,\
        matP, CmatT, \
        CmatTb_trial, \
        CmatTb_test, \
        Nlb_trial,  Nlb_test, \
        trial_func_ndx, trial_func_ndy,  \
        test_func_ndx,  test_func_ndy, \
        matA)
    

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _matA_element_loop(COEFF_FUNC cfunc, \
        int n, int Nb,\
        double[:,:] matP, int[:,:] CmatT, \
        int[:,:] CmatTb_trial, \
        int[:,:] CmatTb_test,  \
        int Nlb_trial,  int Nlb_test, \
        int trial_func_ndx,  int trial_func_ndy,  \
        int test_func_ndx,  int test_func_ndy, \
        double[:,:] matA) nogil:
    # cdef double[2,3] vertMat = matP[:,CmatT[:,n]]
    cdef int vertNum1 = CmatT[0,n]
    cdef int vertNum2 = CmatT[1,n]
    cdef int vertNum3 = CmatT[2,n]
    cdef double x1 = matP[0, vertNum1]
    cdef double y1 = matP[1, vertNum1]
    cdef double x2 = matP[0, vertNum2]
    cdef double y2 = matP[1, vertNum2]
    cdef double x3 = matP[0, vertNum3]
    cdef double y3 = matP[1, vertNum3]
    cdef int alpha=0, beta=0
    cdef double temp = 0.
    for alpha in range(Nlb_trial):
        for beta in range(Nlb_test):
            temp = 1.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _quad_tri_trial_test(COEFF_FUNC cfunc,
                               double x1, double y1,
                               double x2, double y2,
                               double x3, double y3,
                               int Nlb_trial, int Nlb_test,
                               int alpha, int beta,
                               int trial_func_ndx,  int trial_func_ndy,  \
                               int test_func_ndx,  int test_func_ndy, \
                               ) nogil:
    double trial_term = 0.
    double test_term = 0.
    double res = 0.









