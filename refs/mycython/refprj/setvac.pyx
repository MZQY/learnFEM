cimport scipy.special.cython_special as cys
import cython
from cython.parallel import prange
import numpy as np
cimport numpy as np
np.import_array()
from libc.math cimport sqrt, fabs, M_PI, NAN, isnan


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double[:,:] _set_vacuum_psi(double[:] xc1d,\
                                 double[:] yc1d,\
                                 double[:] Ic1d,\
                                 double[:,:] x2d,\
                                 double[:,:] y2d,\
                                 double[:,:] psi2d_nan,\
                                 int[:,:] FRC_bool_int, \
                                 double p1
                                ):
    """
    return psi_vac, i.e. psi value in vacuum
    :
    :param xc1d: (1d np.ndarray) normalized x coord of coils
    :param yc1d: (1d np.ndarray) normalized y coord of coils
    :param Ic1d: (1d np.ndarray) normalized current of coils
    :param x2d: (2d np.ndarray) normalized x coord
    :param y2d: (2d np.ndarray) normalized y coord
    :param psi2d_nan: (2d np.ndarray) psi2d_nan with np.nan in vacuum
    :param FRC_bool_int: (2d np.ndarray) 1 for points in the FRC, 0 for the outside
    :param p1: (float) p1 in normailzed solovev GS equation
    """
    cdef double[:,:] psi2d_vac = np.zeros((psi2d_nan.shape[0], psi2d_nan.shape[1]), dtype=np.double)
    cdef double[:,:] jth2d =  np.zeros((psi2d_nan.shape[0], psi2d_nan.shape[1]), dtype=np.double) # x2d * p1
    cdef int i=0, j=0, k=0
    # cdef double temp_coil_psi = 0.0
    cdef double[:,:] temp_coil_psi = np.zeros((psi2d_nan.shape[0], psi2d_nan.shape[1]), dtype=np.double)

    # initialize jth2d
    for i in range(jth2d.shape[0]):
        for j in range(jth2d.shape[1]):
            jth2d[i,j] = p1 * x2d[i,j]

    for i in prange(psi2d_vac.shape[0], nogil=True):
    # for i in range(psi2d_vac.shape[0]):
        for j in range(psi2d_vac.shape[1]):
            if FRC_bool_int[i, j] == 1:
                psi2d_vac[i,j] = NAN
            else:
                temp_coil_psi[i,j] = 0.0
                for k in range(xc1d.shape[0]):
                    temp_coil_psi[i,j] += _Green(x2d[i,j], y2d[i,j], xc1d[k], yc1d[k]) * Ic1d[k]
                psi2d_vac[i,j] = _intAreaGJ(x2d[i,j], y2d[i,j], x2d, y2d, jth2d, psi2d_nan) + temp_coil_psi[i,j]
    return psi2d_vac



@cython.boundscheck(False)
@cython.wraparound(False)
cdef double _intAreaGJ(double x, double y, \
                      double[:,:] x2d, double[:,:] y2d,\
                      double[:,:] jth2d, double[:,:] psi2d) nogil:
    """
    res = \int \int G(x, y, x_plasma, y_plasma) * Jth(x_plasma, y_plasma) dxdy
    """
    cdef double res = 0.0
    cdef double dx = x2d[1,0] - x2d[0,0]
    cdef double dy = y2d[0,1] - y2d[0,0]
    cdef double dA = dx * dy
    cdef int i=0, j=0
    cdef double xp = 0.0, yp = 0.0
    for i in range(x2d.shape[0]):
        for j in range(x2d.shape[1]):
            if not isnan(psi2d[i,j]):
                xp = x2d[i,j]  # x_plasma
                yp = y2d[i,j]  # y_plasma
                res += _Green(x, y, xp, yp) * jth2d[i,j] * dA
    return res


cdef double _Green(double x, double y, double xp, double yp) nogil:
    """
    Green function
    :param x:  (double) x coord of given point (goal point)
    :param y:  (double) y coord of given point (goal point)
    :param xp: (double) x coord of source point, p for prime
    :param yp: (double) y coord of source point, p for prime
    """
    cdef double res = 0.0
    cdef double k2 = (4.0 * x * xp) / ((x+xp)**2 + (y-yp)**2)
    cdef double k = sqrt(k2)
    cdef double tol = 1.e-14
    if fabs(k) < tol:
        res = 0.0
    elif k2 > (1.0-tol):
        k2 = 1.0-tol
        k = sqrt(k2)
    else:
        res = 1./(M_PI * 2.0) * sqrt(x * xp) / k * ((2-k2) * cys.ellipk(k2) - 2.0 * cys.ellipe(k2))
    return res

def Green(x, y, xp, yp):
    return _Green(x, y, xp, yp)

def intAreaGJ(x, y, x2d, y2d, jth2d, psi2d):
    return _intAreaGJ(x, y, x2d, y2d, jth2d, psi2d)

def set_vacuum_psi(xc1d, yc1d, Ic1d, x2d, y2d, psi2d_nan, FRC_bool, p1):
    cdef np.ndarray FRC_bool_int = np.array(np.where(FRC_bool, 1, 0), dtype=np.int32)
    # print(FRC_bool[:, int(FRC_bool.shape[1]/2)])
    # print(FRC_bool_int[:, int(FRC_bool.shape[1]/2)])
    return np.asarray(_set_vacuum_psi(xc1d, yc1d, Ic1d, x2d, y2d, psi2d_nan, FRC_bool_int, p1))


