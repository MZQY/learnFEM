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


def shape1d_node2(idx, x, node, phi):
    _shape1d_node2(idx, x, node, phi)

def shape1d_node3(idx, x, node, phi):
    _shape1d_node3(idx, x, node, phi)

def shape1d_node4(idx, x, node, phi):
    _shape1d_node4(idx, x, node, phi)

def gauss_legendre_quadrature_set1d(n, x, w):
    _gauss_legendre_quadrature_set1d(n, x, w)

def shape2d_t3(idx, idy, x, y, node_x, node_y, phi):
    _shape2d_t3(idx, idy, x, y, node_x, node_y, phi)


def shape2d_t6(idx, idy, x, y, node_x, node_y, phi):
    _shape2d_t6(idx, idy, x, y, node_x, node_y, phi)


def shape2d_t10(idx, idy, x, y, node_x, node_y, phi):
    _shape2d_t10(idx, idy, x, y, node_x, node_y, phi)

def gauss_quadrature_triangle(n, x, y, w):
    _gauss_quadrature_triangle(n, x, y, w)


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape1d_node2(int idx, double x, double[:] node, double[:] phi) nogil:
    """
    linear shape function on the segment interval [-1, 1]
    2 nodes based on Gauss-Lobatto points
    The values are derived using Mathematica in basis.nb.
    :param idx: (int) derivative order for shape function
    :param x: (double) required coordinate
    :param node: (double[2]) Gauss-Lobatto points
    :param phi: (double[2]) the idx order derivative of shape function on x 
    """
    node[0] = -1.
    node[1] = 1.
    if (x >= -1. and x <= 1.):
        if (idx == 0):
            phi[0] = (1. - x) / 2.
            phi[1] = (1. + x) / 2.
        elif (idx == 1):
            phi[0] = -1. / 2.
            phi[1] =  1. / 2.
        else:
            phi[0] = 0.
            phi[1] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape1d_node3(int idx, double x, double[:] node, double[:] phi) nogil:
    """
    quadratic shape function on the interval [-1, 1]
    3 nodes based on Gauss-Lobatto points
    The values are derived using Mathematica in basis.nb.
    :param idx: (int) derivative order for shape function
    :param x: (double) required coordinate
    :param node: (double[3]) Gauss-Lobatto points
    :param phi: (double[3]) the idx order derivative of shape function on x 
    """
    node[0] = -1.
    node[1] = 0.
    node[2] = 1.
    if (x >= -1. and x <= 1.):
        if (idx == 0):
            phi[0] = 1. / 2. * (-1. + x) * x
            phi[1] = 1. - x * x
            phi[2] = 1. / 2. * x * (1. + x)
        elif (idx == 1):
            phi[0] = -1. / 2. + x
            phi[1] =  -2. * x
            phi[2] = 1. / 2. + x
        elif (idx == 2):
            phi[0] = 1.
            phi[1] = -2.
            phi[2] = 1.
        else:
            phi[0] = 0.
            phi[1] = 0.
            phi[2] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape1d_node4(int idx, double x, double[:] node, double[:] phi) nogil:
    """
    quadratic shape function on the interval [-1, 1]
    4 nodes based on Gauss-Lobatto points
    The values are derived using Mathematica in basis.nb.
    :param idx: (int) derivative order for shape function
    :param x: (double) required coordinate
    :param node: (double[4]) Gauss-Lobatto points
    :param phi: (double[4]) the idx order derivative of shape function on x 
    """
    node[0] = -1.
    node[1] = -sqrt(5.) / 5.
    node[2] =  sqrt(5.) / 5.
    node[3] = 1.
    if (x >= -1. and x <= 1.):
        if (idx == 0):
            phi[0] = 1. / 8. * (-1. + x + 5. * x*x - 5. * x*x*x)
            phi[1] = -1. / 8. * sqrt(5.) * (sqrt(5.) - 5. * x) * (-1. + x) * (1. + x)
            phi[2] = -1. / 8. * sqrt(5.) * (-1. + x) * (1. + x) * (sqrt(5.) + 5. * x)
            phi[3] = 1. / 8. * (-1. - x + 5. * x*x + 5. * x*x*x)
        elif (idx == 1):
            phi[0] = 1. / 8. * (1. + 10. * x - 15. * x*x)
            phi[1] =  -1. / 8. * sqrt(5.) * (5. + 2. * sqrt(5.) * x - 15. * x*x)
            phi[2] = -1. / 8. * sqrt(5.) * (-5. + 2. * sqrt(5.) * x + 15. * x*x)
            phi[3] = 1. / 8. * (-1. + 10. * x + 15. * x*x)
        elif (idx == 2):
            phi[0] = -5. / 4. * (-1. + 3. * x)
            phi[1] = 5. / 4. * (-1. + 3. * sqrt(5.) * x)
            phi[2] = -5. / 4. * (1. + 3. * sqrt(5.) * x)
            phi[3] = 5. / 4. * (1. + 3. * x)
        elif (idx == 3):
            phi[0] = -15. / 4.
            phi[1] = 15. * sqrt(5.) / 4.
            phi[2] = -15. * sqrt(5.) / 4.
            phi[3] = 15. / 4.
        else:
            phi[0] = 0.
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.
        phi[3] = 0.




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _gauss_legendre_quadrature_set1d(int n, \
                                           double[:] x, \
                                           double[:] w) nogil:
    """
    get abscissas and weights for Gauss-Legendre quadrature
    The quadrature interval is [-1, 1]
    Integrate (-1 <= X <= 1) F(x) dX = Sum (1 <= I <= N) W(I) * F(X(I))
    
    The quadrature rule will integrate exactly all polynomials up to x**(2*N-1)

    :param n: (int) number of node points
    :param x: (double[n]) abscissas
    :param w: (double[n]) weights for Gauss-Legendre quadrature

    author: ymma98

    this function is modified from
        void legendre_set ( int n, double x[], double w[] )
        which is created by John Burkardt, the licence of the legendre_set()
        are copied as follows:

        Licensing:
          This code is distributed under the GNU LGPL license. 
        Modified:
          19 October 2009
        Author:
          John Burkardt
        Reference:
          Milton Abramowitz, Irene Stegun,
          Handbook of Mathematical Functions,
          National Bureau of Standards, 1964,
          ISBN: 0-486-61272-4,
          LC: QA47.A34.

          Vladimir Krylov,
          Approximate Calculation of Integrals,
          Dover, 2006,
          ISBN: 0486445798.

          Arthur Stroud, Don Secrest,
          Gaussian Quadrature Formulas,
          Prentice Hall, 1966,
          LC: QA299.4G3S7.

          Daniel Zwillinger, editor,
          CRC Standard Mathematical Tables and Formulae,
          30th Edition,
          CRC Press, 1996,
          ISBN: 0-8493-2479-3.
    """
    if ( n == 1 ):
        x[0] = 0.0
        w[0] = 2.0

    elif ( n == 2 ):
        x[0] = - 0.577350269189625764509148780502
        x[1] =   0.577350269189625764509148780502

        w[0] = 1.0
        w[1] = 1.0

    elif ( n == 3 ):
        x[0] = - 0.774596669241483377035853079956
        x[1] =   0.0
        x[2] =   0.774596669241483377035853079956

        w[0] = 5.0 / 9.0
        w[1] = 8.0 / 9.0
        w[2] = 5.0 / 9.0

    elif ( n == 4 ):
        x[0] = - 0.861136311594052575223946488893
        x[1] = - 0.339981043584856264802665759103
        x[2] =   0.339981043584856264802665759103
        x[3] =   0.861136311594052575223946488893

        w[0] = 0.347854845137453857373063949222
        w[1] = 0.652145154862546142626936050778
        w[2] = 0.652145154862546142626936050778
        w[3] = 0.347854845137453857373063949222

    elif ( n == 5 ):
        x[0] = - 0.906179845938663992797626878299
        x[1] = - 0.538469310105683091036314420700
        x[2] =   0.0
        x[3] =   0.538469310105683091036314420700
        x[4] =   0.906179845938663992797626878299

        w[0] = 0.236926885056189087514264040720
        w[1] = 0.478628670499366468041291514836
        w[2] = 0.568888888888888888888888888889
        w[3] = 0.478628670499366468041291514836
        w[4] = 0.236926885056189087514264040720

    elif ( n == 6 ):
        x[0] = - 0.932469514203152027812301554494
        x[1] = - 0.661209386466264513661399595020
        x[2] = - 0.238619186083196908630501721681
        x[3] =   0.238619186083196908630501721681
        x[4] =   0.661209386466264513661399595020
        x[5] =   0.932469514203152027812301554494

        w[0] = 0.171324492379170345040296142173
        w[1] = 0.360761573048138607569833513838
        w[2] = 0.467913934572691047389870343990
        w[3] = 0.467913934572691047389870343990
        w[4] = 0.360761573048138607569833513838
        w[5] = 0.171324492379170345040296142173
    elif ( n == 7 ):
        x[0] = - 0.949107912342758524526189684048
        x[1] = - 0.741531185599394439863864773281
        x[2] = - 0.405845151377397166906606412077
        x[3] =   0.0
        x[4] =   0.405845151377397166906606412077
        x[5] =   0.741531185599394439863864773281
        x[6] =   0.949107912342758524526189684048

        w[0] = 0.129484966168869693270611432679
        w[1] = 0.279705391489276667901467771424
        w[2] = 0.381830050505118944950369775489
        w[3] = 0.417959183673469387755102040816
        w[4] = 0.381830050505118944950369775489
        w[5] = 0.279705391489276667901467771424
        w[6] = 0.129484966168869693270611432679

    else:
        printf("******************************");
        printf("Illegal value of n = %d \n", n);
        printf("Legal values are 1 through 10");
        printf("******************************");




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape2d_t3(int idx, int idy,  \
                      double x, double y, \
                      double[:] node_x, double[:] node_y, double[:] phi) nogil:
    """
    quadratic shape function on the reference triangle with vertices (0,0), (1,0) and (0,1)
    The values are derived using Mathematica in basis.nb.
     |
     1  3
     |  ..
     |  . .
     y  .  .
     |  .   .
     |  .    .
     0  1-----2
     |
     +--0--x--1-->
    :param idx: (int) derivative order of x for shape function
    :param idy: (int) derivative order of y for shape function
    :param x: (double) required coordinate
    :param y: (double) required coordinate
    :param node_x: (double[3]) coordinate x of nodes
    :param node_y: (double[3]) coordinate y of nodes
    :param phi: (double[3])  shape function after idx order derivative on x and idy order derivative on y
    """
    node_x[0] = 0.   ;  node_y[0] = 0.
    node_x[1] = 1.   ;  node_y[1] = 0.
    node_x[2] = 0.   ;  node_y[2] = 1.
    if ( (x+y-1)<=0. and x >= 0. and y >= 0.):
        if (idx == 0 and idy == 0):
            phi[0] = 1. - x - y
            phi[1] = x
            phi[2] = y
        elif (idx == 1 and idy == 0):
            phi[0] = -1.
            phi[1] = 1.
            phi[2] = 0.
        elif (idx == 0 and idy == 1):
            phi[0] = -1.
            phi[1] = 0.
            phi[2] = 1.
        else:
            phi[0] = 0.
            phi[1] = 0.
            phi[2] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.



@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape2d_t6(int idx, int idy,  \
                      double x, double y, \
                      double[:] node_x, double[:] node_y, double[:] phi) nogil:
    """
    quadratic shape function on the reference triangle with vertices (0,0), (1,0) and (0,1)
    The values are derived using Mathematica in basis.nb.
     |
     1  3
     |  ..
     |  . .
     y  5  4
     |  .   .
     |  .    .
     0  1--6--2
     |
     +--0--x--1-->
    :param idx: (int) derivative order of x for shape function
    :param idy: (int) derivative order of y for shape function
    :param x: (double) required coordinate
    :param y: (double) required coordinate
    :param node_x: (double[6]) coordinate x of nodes
    :param node_y: (double[6]) coordinate y of nodes
    :param phi: (double[6])  shape function after idx order derivative on x and idy order derivative on y
    """
    node_x[0] = 0.       ;  node_y[0] = 0.
    node_x[1] = 1.       ;  node_y[1] = 0.
    node_x[2] = 0.       ;  node_y[2] = 1.
    node_x[3] = 1. / 2.  ;  node_y[3] = 1. / 2.
    node_x[4] = 0.       ;  node_y[4] = 1. / 2.
    node_x[5] = 1. / 2.  ;  node_y[5] = 0.
    if ( (x+y-1) <= 0. and x >= 0. and y >= 0.):
        if (idx == 0 and idy == 0):
            phi[0] = 1. + 2. * x*x - 3. * y + 2. * y*y + x * (-3. + 4.* y)
            phi[1] = x * (-1. + 2. * x)
            phi[2] = y * (-1. + 2. * y)
            phi[3] = 4. * x * y
            phi[4] = -4. * y * (-1. + x + y)
            phi[5] = -4. * x * (-1. + x + y)
        elif (idx == 1 and idy == 0):
            phi[0] = -3. + 4. * x + 4. * y
            phi[1] = -1. + 4. * x
            phi[2] = 0.
            phi[3] = 4. * y
            phi[4] = -4. * y
            phi[5] = -4. * (-1. + 2. * x + y)
        elif (idx == 0 and idy == 1):
            phi[0] = -3. + 4. * x + 4. * y
            phi[1] = 0.
            phi[2] = -1. + 4. * y
            phi[3] = 4. * x
            phi[4] = -4. * (-1. + x + 2. * y)
            phi[5] = -4. * x
        elif (idx == 1 and idy == 1):
            phi[0] = 4.
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = 4.
            phi[4] = -4.
            phi[5] = -4.
        elif (idx == 2 and idy == 0):
            phi[0] = 4.
            phi[1] = 4.
            phi[2] = 0.
            phi[3] = 0.
            phi[4] = 0.
            phi[5] = -8.
        elif (idx == 0 and idy == 2):
            phi[0] = 4.
            phi[1] = 0.
            phi[2] = 4.
            phi[3] = 0.
            phi[4] = -8.
            phi[5] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.
        phi[3] = 0.
        phi[4] = 0.
        phi[5] = 0.




@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _shape2d_t10(int idx, int idy,  \
                       double x, double y, \
                       double[:] node_x, double[:] node_y, double[:] phi) nogil:
    """
    quadratic shape function on the reference triangle with vertices (0,0), (1,0) and (0,1)
    The values are derived using Mathematica in basis.nb.
     |
     1  3
     |  ..
     |  . .
     |  7  5
     y  .   .
     |  .    .
     |  6 10  4
     |  .      .
     0  1--8--9-2
     |
     +--0--x--1-->
    :param idx: (int) derivative order of x for shape function
    :param idy: (int) derivative order of y for shape function
    :param x: (double) required coordinate
    :param y: (double) required coordinate
    :param node_x: (double[10]) coordinate x of nodes
    :param node_y: (double[10]) coordinate y of nodes
    :param phi: (double[10])  shape function after idx order derivative on x and idy order derivative on y
    """
    node_x[0] = 0.       ;  node_y[0] = 0.
    node_x[1] = 1.       ;  node_y[1] = 0.
    node_x[2] = 0.       ;  node_y[2] = 1.
    node_x[3] = 2. / 3.  ;  node_y[3] = 1. / 3.
    node_x[4] = 1. / 3.  ;  node_y[4] = 2. / 3.
    node_x[5] = 0.       ;  node_y[5] = 1. / 3.
    node_x[6] = 0.       ;  node_y[6] = 2. / 3.
    node_x[7] = 1. / 3.  ;  node_y[7] = 0.
    node_x[8] = 2. / 3.  ;  node_y[8] = 0.
    node_x[9] = 1. / 3.  ;  node_y[9] = 1. / 3.
    if ( (x+y-1)<=0. and x >= 0. and y >=0.):
        if (idx == 0 and idy == 0):
            phi[0] = (2. - 9. * x*x*x - 11. * y + 18.* y*y - 9.* y*y*y - 9. * x*x *(-2. + 3. * y) + x * (-11. + 36. * y - 27. * y*y))/2.
            phi[1] = (x * (2. - 9. * x + 9. * x*x ))/2.
            phi[2] = (y * (2. - 9. * y + 9. * y*y ))/2.
            phi[3] = (9.* x * (-1. + 3. * x ) * y)/2.
            phi[4] = (9. * x * y * (-1. + 3. * y))/2.
            phi[5] = (9. * y * (2. + 3.* x*x - 5. * y + 3 * y*y + x * (-5. + 6 * y) ))/2.
            phi[6] = (-9. * y * (-1. + x + y) * (-1. + 3. * y))/2.
            phi[7] = (9. * x * (2. + 3. * x*x - 5. * y + 3. * y*y + x * (-5. + 6. * y)))/2.
            phi[8] = (-9. * x * (-1. + 3. * x) * (-1. + x + y))/2.
            phi[9] = -27. * x * y * (-1. + x + y)
        elif (idx == 1 and idy == 0):
            phi[0] = (-11. - 27. * x*x + x * (36. - 54. * y) + 36. * y - 27. * y*y)/2.
            phi[1] = 1. - 9. * x + (27. * x*x )/2.
            phi[2] = 0.
            phi[3] = (9. * (-1. + 6. * x) * y)/2.
            phi[4] = (9. * y * (-1. + 3. * y ))/2.
            phi[5] = (9. * y * (-5. + 6. * x + 6. * y))/2.
            phi[6] = (9. * (1. - 3. * y ) * y)/2.
            phi[7] = (9. * (2. + 9. * x*x - 5. * y + 3. * y*y + 2. * x * (-5. + 6. * y)))/2.
            phi[8] = (-9. * (1. + 9. * x*x - y + x * (-8. + 6. * y)))/2.
            phi[9] = -27. * y * (-1. + 2. * x + y)
        elif (idx == 0 and idy == 1):
            phi[0] = (-11. - 27. * x*x + x * (36. - 54. * y) + 36. * y - 27. * y*y)/2.
            phi[1] = 0.
            phi[2] = 1. - 9. * y + (27. * y*y)/2.
            phi[3] = (9. * x * (-1. + 3. * x ))/2.
            phi[4] = (9. * x * (-1. + 6. * y ))/2.
            phi[5] = (9. * (2. + 3. * x*x - 10. * y + 9. * y*y + x * (-5. + 12. * y)))/2.
            phi[6] = (-9. * (1. - 8. * y + 9. * y*y + x * (-1. + 6. * y)))/2.
            phi[7] = (9. * x * (-5. + 6. * x + 6. * y))/2.
            phi[8] = (9. * (1. - 3. * x) * x)/2.
            phi[9] = -27. * x * (-1. + x + 2. * y)
        elif (idx == 1 and idy == 1):
            phi[0] = -9. * (-2. + 3. * x + 3. * y)
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = -4.5 + 27. * x
            phi[4] = -4.5 + 27. * y
            phi[5] = (9. * (-5. + 6. * x + 12. * y))/2.
            phi[6] = 4.5 - 27. * y
            phi[7] = (9. * (-5. + 12. * x + 6. * y))/2.
            phi[8] = 4.5 - 27. * x
            phi[9] = -27. * (-1. + 2. * x + 2. * y)
        elif (idx == 2 and idy == 0):
            phi[0] = -9. * (-2. + 3. * x + 3. * y)
            phi[1] = -9. + 27. * x
            phi[2] = 0.
            phi[3] = 27. * y
            phi[4] = 0.
            phi[5] = 27. * y
            phi[6] = 0.
            phi[7] = 9. * (-5. + 9. * x + 6. * y)
            phi[8] = -9. * (-4. + 9. * x + 3. * y)
            phi[9] = -54. * y
        elif (idx == 0 and idy == 2):
            phi[0] = -9. * (-2. + 3. * x + 3. * y)
            phi[1] = 0.
            phi[2] = -9. + 27. * y
            phi[3] = 0.
            phi[4] = 27. * x
            phi[5] = 9. * (-5. + 6. * x + 9. * y)
            phi[6] = -9. * (-4. + 3. * x + 9. * y)
            phi[7] = 27. * x
            phi[8] = 0.
            phi[9] = -54. * x
        elif (idx == 1 and idy == 2):
            phi[0] = -27.
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = 0.
            phi[4] = 27.
            phi[5] = 54.
            phi[6] = -27.
            phi[7] = 27.
            phi[8] = 0.
            phi[9] = -54.
        elif (idx == 2 and idy == 1):
            phi[0] = -27.
            phi[1] = 0.
            phi[2] = 0.
            phi[3] = 27.
            phi[4] = 0.
            phi[5] = 27.
            phi[6] = 0.
            phi[7] = 54.
            phi[8] = -27.
            phi[9] = -54.
        elif (idx == 3 and idy == 0):
            phi[0] = -27.
            phi[1] = 27.
            phi[2] = 0.
            phi[3] = 0.
            phi[4] = 0.
            phi[5] = 0.
            phi[6] = 0.
            phi[7] = 81.
            phi[8] = -81.
            phi[9] = 0.
        elif (idx == 0 and idy == 3):
            phi[0] = -27.
            phi[1] = 0.
            phi[2] = 27.
            phi[3] = 0.
            phi[4] = 0.
            phi[5] = 81.
            phi[6] = -81.
            phi[7] = 0.
            phi[8] = 0.
            phi[9] = 0.
    else:
        phi[0] = 0.
        phi[1] = 0.
        phi[2] = 0.
        phi[3] = 0.
        phi[4] = 0.
        phi[5] = 0.
        phi[6] = 0.
        phi[7] = 0.
        phi[8] = 0.
        phi[9] = 0.





@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _gauss_quadrature_triangle(int n, \
                                    double[:] x, \
                                    double[:] y,
                                    double[:] w) nogil:
    """
    get Gaussian quadrature points and weights for the reference triangle
      ^
    1 | *
      | ..
    Y | . .
      | .  .
    0 | *---*
      +------->
       0 X 1
    Integrate F(x, y) dX dY = Area(T) * Sum (1 <= I <= N) W(I) * F(X(I), Y(I)), 
    Area(T) = 1/2 here
    ref: https://people.sc.fsu.edu/~jburkardt/datasets/quadrature_rules_tri/quadrature_rules_tri.html
    
    :param n: (int) number of node points
    :param x: (double[n]) x coordinate of the quadrature point
    :param y: (double[n]) y coordinate of the quadrature point
    :param w: (double[n]) weights for Gauss-Legendre quadrature

    author: ymma98

    this function is modified from
        void triangle_unit_set ( int rule, double xtab[], double ytab[], double weight[] )
        which is created by John Burkardt

    references:
    H R Schwarz,
    Methode der Finiten Elemente,
    Teubner Studienbuecher, 1980.

    Strang and Fix,
    An Analysis of the Finite Element Method,
    Prentice Hall, 1973, page 184.

    Arthur H Stroud,
    Approximate Calculation of Multiple Integrals,
    Prentice Hall, 1971.

    O C Zienkiewicz,
    The Finite Element Method,
    McGraw Hill, Third Edition, 1977, page 201.

    """
    # 3 points, precision 2, Strang and Fix formula #2.
    if (n == 3):
        x[0] = 0.0
        x[1] = 1.0 / 2.0
        x[2] = 1.0 / 2.0

        y[0] = 1.0 / 2.0
        y[1] = 0.0
        y[2] = 1.0 / 2.0

        w[0] = 1.0 / 3.0
        w[1] = 1.0 / 3.0
        w[2] = 1.0 / 3.0
    # 4 points, precision 3, Strang and Fix formula #3
    elif (n == 4):
        x[0] = 10.0 / 30.0
        x[1] = 18.0 / 30.0
        x[2] =  6.0 / 30.0
        x[3] =  6.0 / 30.0

        y[0] = 10.0 / 30.0
        y[1] =  6.0 / 30.0
        y[2] = 18.0 / 30.0
        y[3] =  6.0 / 30.0

        w[0] = -27.0 / 48.0
        w[1] =  25.0 / 48.0
        w[2] =  25.0 / 48.0
        w[3] =  25.0 / 48.0
    #  6 points, precision 4, Strang and Fix, formula #5
    elif (n==6):
        x[0] = 0.816847572980459
        x[1] = 0.091576213509771
        x[2] = 0.091576213509771
        x[3] = 0.108103018168070
        x[4] = 0.445948490915965
        x[5] = 0.445948490915965

        y[0] = 0.091576213509771
        y[1] = 0.816847572980459
        y[2] = 0.091576213509771
        y[3] = 0.445948490915965
        y[4] = 0.108103018168070
        y[5] = 0.445948490915965

        w[0] = 0.109951743655322
        w[1] = 0.109951743655322
        w[2] = 0.109951743655322
        w[3] = 0.223381589678011
        w[4] = 0.223381589678011
        w[5] = 0.223381589678011
    # 9 points, precision 6, Strang and Fix formula #8.
    elif (n==9):
        x[0] = 0.124949503233232
        x[1] = 0.437525248383384
        x[2] = 0.437525248383384
        x[3] = 0.797112651860071
        x[4] = 0.797112651860071
        x[5] = 0.165409927389841
        x[6] = 0.165409927389841
        x[7] = 0.037477420750088
        x[8] = 0.037477420750088

        y[0] = 0.437525248383384
        y[1] = 0.124949503233232
        y[2] = 0.437525248383384
        y[3] = 0.165409927389841
        y[4] = 0.037477420750088
        y[5] = 0.797112651860071
        y[6] = 0.037477420750088
        y[7] = 0.797112651860071
        y[8] = 0.165409927389841

        w[0] = 0.205950504760887
        w[1] = 0.205950504760887
        w[2] = 0.205950504760887
        w[3] = 0.063691414286223
        w[4] = 0.063691414286223
        w[5] = 0.063691414286223
        w[6] = 0.063691414286223
        w[7] = 0.063691414286223
        w[8] = 0.063691414286223

    else:
        printf("******************************")
        printf("Illegal value of n = %d \n", n)
        printf("Legal values are 3, 4, 6, 9")
        printf("******************************")
