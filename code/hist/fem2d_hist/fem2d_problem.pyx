"""
FEM basis functions and Gaussian quadrature
ymma98
"""

import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt, fabs
from libc.stdio cimport printf
from libc.stdlib cimport exit

include "./fem2d.pyx"


def pget_matA(N, Nb,\
        matP, CmatT, \
        CmatTb_trial, \
        CmatTb_test,  \
        Nlb_trial, Nlb_test, \
        trial_func_ndx, trial_func_ndy,  \
        test_func_ndx, test_func_ndy, \
        matA):
    get_matA(N, Nb,\
        matP, CmatT, \
        CmatTb_trial, \
        CmatTb_test,  \
        Nlb_trial, Nlb_test, \
        trial_func_ndx, trial_func_ndy,  \
        test_func_ndx, test_func_ndy, \
        matA)

def pget_vecb(N, Nb,\
        matP, CmatT, \
        CmatTb_test,  \
        Nlb_test, \
        test_func_ndx, test_func_ndy, \
        vecb):
    get_vecb(N, Nb,\
        matP, CmatT, \
        CmatTb_test,  \
        Nlb_test, \
        test_func_ndx, test_func_ndy, \
        vecb)


cdef double coeff_func1(double x, double y) nogil:
    return 2.

cdef double coeff_func2(double x, double y) nogil:
    return 2.


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_matA(int N, int Nb,\
        double[:,:] matP, int[:,:] CmatT, \
        int[:,:] CmatTb_trial, \
        int[:,:] CmatTb_test,  \
        int Nlb_trial,  int Nlb_test, \
        int trial_func_ndx,  int trial_func_ndy,  \
        int test_func_ndx,  int test_func_ndy, \
        double[:,:] matA) nogil:
    """
    :param N: (int) number of mesh element
    :param Nb: (int) number of FE basis functions, which is also the number of FE nodes
    :param matP: (2d double matrix) mesh coordinate matrix
    :param CmatT: (2d int matrix) mesh element node number matrix, index starts from 0 (C form)
    :param CmatTb_trial: (2d int matrix) FE node number matrix for trial functions, index starts from 0 (C form)
    :param CmatTb_test: (2d int matrix) FE node number matrix for test functions, index starts from 0 (C form)
    :param Nlb_trial: (int) trial basis function number on a single mesh element, which is also the number of FE nodes on the single mesh element
    :param Nlb_test: (int) test basis function number on a single mesh element, which is also the number of FE nodes on the single mesh element
    :param trial_func_ndx: (int) derivative order in the x-dir for trial function
    :param trial_func_ndy: (int) derivative order in the y-dir for trial function
    :param test_func_ndx: (int) derivative order in the x-dir for test function
    :param test_func_ndy: (int) derivative order in the y-dir for test function
    :param matA: (2d double matrix [Nb, Nb]) stiffness matrix
    """
    _get_matA (coeff_func1, \
    N, Nb,\
    matP, CmatT, \
    CmatTb_trial, \
    CmatTb_test,  \
    Nlb_trial, Nlb_test, \
    trial_func_ndx, trial_func_ndy,  \
    test_func_ndx, test_func_ndy, \
    matA)



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void get_vecb (int N, int Nb,\
        double[:,:] matP, int[:,:] CmatT, \
        int[:,:] CmatTb_test,  \
        int Nlb_test, \
        int test_func_ndx,  int test_func_ndy, \
        double[:] vecb) nogil:
    """
    :param cfunc: (cython function) coefficient function
    :param N: (int) number of mesh element
    :param Nb: (int) number of FE basis functions, which is also the number of FE nodes
    :param matP: (2d double matrix) mesh coordinate matrix
    :param CmatT: (2d int matrix) mesh element node number matrix, index starts from 0 (C form)
    :param CmatTb_test: (2d int matrix) FE node number matrix for test functions, index starts from 0 (C form)
    :param Nlb_test: (int) test basis function number on a single mesh element, which is also the number of FE nodes on the single mesh element
    :param test_func_ndx: (int) derivative order in the x-dir for test function
    :param test_func_ndy: (int) derivative order in the y-dir for test function
    :param matA: (2d double matrix [Nb, Nb]) stiffness matrix
    """
    _get_vecb (coeff_func2, \
    N, Nb,\
    matP, CmatT, \
    CmatTb_test,  \
    Nlb_test, \
    test_func_ndx, test_func_ndy, \
    vecb)

