import numpy as np
import findiff
import matplotlib.pyplot as plt
import scipy.interpolate


def check_GS(p1, x2d_norm, y2d_norm, psi2d_norm):
    """
    lhs = \Delta^* psi
    rhs = -x^2 p1
    """
    dx = x2d_norm[1,0] - x2d_norm[0,0]
    dy = y2d_norm[0,1] - y2d_norm[0,0]
    dr = findiff.FinDiff(0, dx, 1)
    dr2 = findiff.FinDiff(0, dx, 2)
    dz2 = findiff.FinDiff(1, dy, 2)
    dpsidr = dr(psi2d_norm)
    dpsidr2 = dr2(psi2d_norm)
    dpsidz2 = dz2(psi2d_norm)
    lhs = np.zeros(x2d_norm.shape)
    rhs = -x2d_norm**2 * p1
    # rhs = np.zeros(x2d_norm.shape)
    for i in range(x2d_norm.shape[0]):
        for j in range(x2d_norm.shape[1]):
            if x2d_norm[i, j] == 0:
                lhs[i,j] = rhs[i,j]
            else:
                lhs[i,j] = dpsidz2[i,j]+ dpsidr2[i,j] - dpsidr[i,j]/x2d_norm[i,j]
    err = lhs - rhs
    # err = np.where(np.isnan(err), 0, err)
    abserr = np.abs(err)
    maxabslhs = np.amax(np.abs(lhs))
    maxabsrhs = np.amax(np.abs(rhs))
    print("*****"*5)
    print("*** error calculation of the GS equation ***")
    print("max abs(lhs) = {}".format(maxabslhs))
    print("max abs(rhs) = {}".format(maxabsrhs))
    print("max(|err|)/max(|lhs|) = {:.3e}%".format(np.amax(abserr)/maxabslhs*100))
    print("*****"*5)
    # fig = plt.figure(num=333)
    # ax = fig.add_subplot(111)
    # cs = ax.contourf(x2d_norm, y2d_norm, err)
    # fig.colorbar(cs)



def check_GS_af_match(p1, x2d_norm, y2d_norm, psi2d_tot, FRC_bool):
    """
    check psi after matching satisfy GS equation or not
    lhs = \Delta^* psi
    rhs = -x^2 p1
    rhs = 0 in vacuum
    """
    dx = x2d_norm[1,0] - x2d_norm[0,0]
    dy = y2d_norm[0,1] - y2d_norm[0,0]
    dr = findiff.FinDiff(0, dx, 1, acc=4)
    dr2 = findiff.FinDiff(0, dx, 2, acc=4)
    dz2 = findiff.FinDiff(1, dy, 2, acc=4)
    dpsidr = dr(psi2d_tot)
    dpsidr2 = dr2(psi2d_tot)
    dpsidz2 = dz2(psi2d_tot)
    lhs = np.zeros(x2d_norm.shape)
    rhs = -x2d_norm**2 * p1
    rhs = np.where(FRC_bool, rhs, 0.0)
    # rhs = np.zeros(x2d_norm.shape)
    for i in range(x2d_norm.shape[0]):
        for j in range(x2d_norm.shape[1]):
            if x2d_norm[i, j] == 0:
                lhs[i,j] = rhs[i,j]
            else:
                lhs[i,j] = dpsidz2[i,j]+ dpsidr2[i,j] - dpsidr[i,j]/x2d_norm[i,j]
    err = lhs - rhs
    # err = np.where(np.isnan(err), 0, err)
    abserr = np.abs(err)
    maxabslhs = np.amax(np.abs(lhs))
    maxabsrhs = np.amax(np.abs(rhs))
    print("*****"*5)
    print("*** error calculation of the vacuum GS equation ***")
    print("max abs(lhs) = {}".format(maxabslhs))
    print("max abs(rhs) = {}".format(maxabsrhs))
    print("max(|err|)/max(|lhs|) = {:.3e}%".format(np.amax(abserr)/maxabslhs*100))
    print("*****"*5)
    fig = plt.figure(num=333)
    ax = fig.add_subplot(121)
    cs = ax.contourf(x2d_norm, y2d_norm, abserr)

    Xflat, Yflat, Zflat = x2d_norm.flatten(), y2d_norm.flatten(), err.flatten()
    def fmt(x, y):
        # get closest point with known data
        dist = np.linalg.norm(np.vstack([Xflat - x, Yflat - y]), axis=0)
        idx = np.argmin(dist)
        z = Zflat[idx]
        return 'x={x:.5f}  y={y:.5f}  z={z:.5f}'.format(x=x, y=y, z=z)

    ax.format_coord = fmt

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(x2d_norm, y2d_norm, abserr, cmap='rainbow')

    fig.colorbar(cs)
