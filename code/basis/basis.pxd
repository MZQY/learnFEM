"""
FEM basis functions and Gaussian quadrature
ymma98
"""

import cython


cdef double _affinemap1d_xhat_to_x(double x1, double x2, double xhat) nogil


cdef void _shape1d_node2(int idx, double x, double[2] node, double[2] phi) nogil

cdef void _shape1d_node3(int idx, double x, double[3] node, double[3] phi) nogil



cdef void _shape1d_node4(int idx, double x, double[4] node, double[4] phi) nogil

cdef void _gauss_legendre_quadrature_set1d(int n, \
                                           double[:] x, \
                                           double[:] w) nogil

cdef void _shape2d_t3(int idx, int idy,  \
                      double x, double y, \
                      double[3] node_x, double[3] node_y, double[3] phi) nogil



cdef void _shape2d_t6(int idx, int idy,  \
                      double x, double y, \
                      double[6] node_x, double[6] node_y, double[6] phi) nogil


cdef void _shape2d_t10(int idx, int idy,  \
                       double x, double y, \
                       double[10] node_x, double[10] node_y, double[10] phi) nogil

cdef void _gauss_quadrature_triangle(int n, \
                                    double[:] x, \
                                    double[:] y, \
                                    double[:] w) nogil
