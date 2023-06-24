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

include "../basis/pbasis2d.pyx"


cdef double coeff_function(double x, double y):
    return 2.


# https://stackoverflow.com/questions/18348083/passing-cython-function-to-cython-function
ctypedef double (*COEFF_FUNC)(double x, double y) nogil


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
    """
    :param cfunc: (cython function) coefficient function
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

    cdef int vertNum1 = CmatT[0,n]
    cdef int vertNum2 = CmatT[1,n]
    cdef int vertNum3 = CmatT[2,n]
    cdef double x1 = matP[0, vertNum1]
    cdef double y1 = matP[1, vertNum1]
    cdef double x2 = matP[0, vertNum2]
    cdef double y2 = matP[1, vertNum2]
    cdef double x3 = matP[0, vertNum3]
    cdef double y3 = matP[1, vertNum3]
    cdef int alpha=0, beta=0, i, j
    cdef double temp = 0.
    for alpha in range(Nlb_trial):
        for beta in range(Nlb_test):
            temp = _quad_tri_trial_test(cfunc,
                               x1, y1,
                               x2, y2,
                               x3, y3,
                               Nlb_trial, Nlb_test,
                               alpha, beta,
                               trial_func_ndx,  trial_func_ndy,  \
                               test_func_ndx,  test_func_ndy)
            i = CmatTb_test[beta, n]
            j = CmatTb_trial[beta, n]
            matA[i,j] += temp


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
    """
    :param cfunc: (cython function) coefficient function
    :param x1, y1, x2, y2, x3, y3: (double) coordinates of the local triangle vertices
    :param Nlb_trial: (int) trial basis function number on a single mesh element, which is also the number of FE nodes on the single mesh element
    :param Nlb_test: (int) test basis function number on a single mesh element, which is also the number of FE nodes on the single mesh element
    :param alpha: (int) index for trial local basis functions
    :param beta: (int) index for test local basis functions
    :param trial_func_ndx: (int) derivative order of trial function in x-dir
    :param trial_func_ndy: (int) derivative order of trial function in y-dir
    :param test_func_ndx: (int) derivative order of test function in x-dir
    :param test_func_ndy: (int) derivative order of test function in y-dir
    """
    cdef int gauss_quad_num = 9
    cdef double[9] gauss_quad_refx
    cdef double[9] gauss_quad_refy
    cdef double[9] gauss_quad_refw
    cdef int i=0, j=0
    cdef double trial_ref_00, trial_ref_01, trial_ref10, trial_ref11, trial_ref20, trial_ref02
    cdef double test_ref_00, test_ref_01, test_ref10, test_ref11, test_ref20, test_ref02
    cdef double temp_loc_x
    cdef double temp_loc_y
    cdef double dux_dx, dux_dy, duy_dx, duy_dy
    cdef double jfac = 0.
    cdef double res = 0.


    # get gauss points and weights on ref triangle
    _gauss_quad_triangle_9p(gauss_quad_refx, gauss_quad_refy, gauss_quad_refw)

    # calculate the jacobian factor det(J)
    jfac = fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    
    # calc derivative term due to integration by part
    dux_dx = (y3 - y1) / jfac
    dux_dy = (x1 - x3) / jfac
    duy_dx = (y1 - y2) / jfac
    duy_dy = (x2 - x1) / jfac


    for i in range(gauss_quad_num):
        temp_loc_x = gauss_quad_refx[i] * (x2 - x1) + gauss_quad_refy[i] * (x3 - x1) + x1
        temp_loc_y = gauss_quad_refx[i] * (y2 - y1) + gauss_quad_refy[i] * (y3 - y1) + y1
        
        res += cfunc(temp_loc_x, temp_loc_y) * \
                _shape2d_ref_include_int_by_part(Nlb_trial, alpha, trial_func_ndx, trial_func_ndy,\
                                                 gauss_quad_refx[i], gauss_quad_refy[i], \
                                                 dux_dx, dux_dy, \
                                                 duy_dx, duy_dy)  * \
                _shape2d_ref_include_int_by_part(Nlb_test, beta, test_func_ndx, test_func_ndy,\
                                                 gauss_quad_refx[i], gauss_quad_refy[i], \
                                                 dux_dx, dux_dy, \
                                                 duy_dx, duy_dy)
    
    res = res * 0.5 * jfac
    
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _shape2d_ref_include_int_by_part(int Nlb, int basis_id, \
                                             int idx, int idy,\
                                             double x, double y, \
                                             double dux_dx, double dux_dy,
                                             double duy_dx, double duy_dy
                                             ) nogil:
    cdef double res = 0.
    if (idx == 0 and idy ==0):
        res = _shape2d_ref(Nlb, basis_id, 0, 0, x, y)
    elif (idx == 1 and idy == 0):
        res = _shape2d_ref(Nlb, basis_id, 1, 0, x, y) * dux_dx + \
              _shape2d_ref(Nlb, basis_id, 0, 1, x, y) * duy_dx 
    elif (idx == 0 and idy == 1):
        res = _shape2d_ref(Nlb, basis_id, 1, 0, x, y) * dux_dy + \
              _shape2d_ref(Nlb, basis_id, 0, 1, x, y) * duy_dy 
    elif (idx == 1 and idy == 1):
        res = _shape2d_ref(Nlb, basis_id, 1, 1, x, y) * (duy_dy * dux_dx + dux_dy * duy_dx) + \
              _shape2d_ref(Nlb, basis_id, 2, 0, x, y) * dux_dy * dux_dx + \
              _shape2d_ref(Nlb, basis_id, 0, 2, x, y) * duy_dy * duy_dx
    elif (idx == 2 and idy == 0):
        res = _shape2d_ref(Nlb, basis_id, 1, 1, x, y) * 2. * duy_dx * dux_dx + \
              _shape2d_ref(Nlb, basis_id, 2, 0, x, y) * dux_dx * dux_dx + \
              _shape2d_ref(Nlb, basis_id, 0, 2, x, y) * duy_dx * duy_dx
    elif (idx == 0 and idy == 2):
        res = _shape2d_ref(Nlb, basis_id, 1, 1, x, y) * 2. * duy_dy * dux_dy + \
              _shape2d_ref(Nlb, basis_id, 2, 0, x, y) * dux_dy * dux_dy + \
              _shape2d_ref(Nlb, basis_id, 0, 2, x, y) * duy_dy * duy_dy
    else:
        printf("******************************");
        printf("(idx, idy) can only be (0,0), (0,1), (1,0), (1,1), (2,0) and (0,2)");
        printf("******************************");
        exit(1)
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _shape2d_ref(int Nlb, int basis_id,\
                       int idx, int idy,  \
                       double x, double y) nogil:
    """
    :param Nlb: (int) basis function number on a single mesh element, which is also the number of FE nodes on the single mesh element
    :param basis_id: (int) index for local basis functions, 0 <= basis_id <= Nlb-1
    :param idx: (int) derivative order in x-dir
    :param idy: (int) derivative order in y-dir
    :param x: (double) x coordinate in reference triangle
    :param y: (double) y coordinate in reference triangle
    """
    cdef double[3] t3node_x, t3node_y, phi_t3
    cdef double[6] t6node_x, t6node_y, phi_t6
    cdef double[10] t10node_x, t10node_y, phi_t10
    cdef double res = 0.

    if Nlb == 3:
        _shape2d_t3(idx, idy, x, y, t3node_x, t3node_y, phi_t3)
        res = phi_t3[basis_id]
    elif Nlb == 6:
        _shape2d_t6(idx, idy, x, y, t6node_x, t6node_y, phi_t6)
        res = phi_t6[basis_id]
    elif Nlb == 10:
        _shape2d_t10(idx, idy, x, y, t10node_x, t10node_y, phi_t10)
        res = phi_t10[basis_id]
    else:
        printf("******************************");
        printf("Nlb can only be 3 or 6 or 10 for the 2d triangle FE element.");
        printf("******************************");
        exit(1)
    return res


