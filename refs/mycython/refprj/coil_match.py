import numpy as np
import matplotlib.pyplot as plt
from setvac import *

# find points within separatrix
# from shapely.geometry import Point
# from shapely.geometry.polygon import Polygon

def nan_psi_out_separatrix(Xp_y_norm,\
                           x2d_norm, \
                           y2d_norm, \
                           psi2d):
    """
    :param Xp_y_norm: (float) normalized y coord of X point
    :param x2d_norm: (2d np.ndarray) normalized x2d
    :param y2d_norm: (2d np.ndarray) normalized y2d
    :param psi2d: (2d np.ndarray) psi
    """
    dx = x2d_norm[1,0] - x2d_norm[0,0]
    dy = y2d_norm[0,1] - y2d_norm[0,0]
    # conserve core region valid, inside rs and Xp_y_norm
    psi2d_nan = np.where(np.abs(y2d_norm)>Xp_y_norm+dy*2,\
                     np.nan, psi2d)
    psi2d_nan = np.where(x2d_norm>np.sqrt(2)+dx*2, np.nan, psi2d_nan)

    # get coordinate of the separatrix
    temp_fig = plt.figure(num=2333)
    temp_ax = temp_fig.add_subplot(111)
    temp_cs = temp_ax.contour(x2d_norm, y2d_norm, psi2d_nan, [0.0])
    sep_coord = temp_cs.allsegs[0][0]
    x_sep = sep_coord[:,0]
    y_sep = sep_coord[:,1]
    temp_fig.clear()
    plt.close(temp_fig)
    del temp_cs, sep_coord
    sep_points = np.column_stack((x_sep, y_sep))

    psi2d_nan = np.where(psi2d_nan<np.abs(np.nanmax(psi2d))*np.amin([dx, dy])**4, psi2d_nan, np.nan)

    FRC_bool = np.where(np.isnan(psi2d_nan), False, True)

    print('\n'); print("*****"*5)
    print("max x for FRC_bool:")
    print(np.nanmax(np.where(FRC_bool, x2d_norm, np.nan)))
    print("max x for sep_points:")
    print(np.amax(sep_points[:,0]))
    print("*****"*5); print("\n")
    return FRC_bool, sep_points


def get_uniform_points_on_ellipse(nmatch):
    # https://math.stackexchange.com/questions/2093569/points-on-an-ellipse
    Xp_y = 7.07
    if Xp_y > np.sqrt(2):
        t = np.linspace(np.pi * 5/180, np.pi/2, nmatch)  # [0, 80 deg], uniform
        e = np.sqrt(1.0 - np.sqrt(2)**2/Xp_y**2)
    else:
        t = np.linspace(0, np.pi * 80/180, nmatch)  # [0, 80 deg], uniform
        e = np.sqrt(1.0 - Xp_y**2/np.sqrt(2)**2)
    theta = t + (e**2/8 + e**4/16 + 71*e**6/2048) * np.sin(2*t) + (5*e**4/256 + 5*e**6/256) * np.sin(4*t) + (29*e**6/6144)*np.sin(6*t)
    if Xp_y > np.sqrt(2):
        xp = Xp_y * np.cos(theta)
        yp = np.sqrt(2) * np.sin(theta)
        xp, yp = yp, xp
    else:
        xp = np.sqrt(2) * np.cos(theta)
        yp = Xp_y * np.sin(theta)
    return xp


def get_match_points(ncoil, sep_points, mul_ncoil_coeff=2, print_length_flag=False):
    """
    :param ncoil: (int) coil number in the upper plane, assuming up-down symmetry
    :param sep_points: (2d np.ndarray) points on the contour
    :param mul_ncoil_coeff: (int) number of matching points = mul_ncoil_coeff * ncoil
    :param print_length_flag: (bool) if True, print length of array positive_points and match_points
    """
    positive_points = sep_points[np.where(sep_points[:,1] > 0)]
    ## equal distance for x
    positive_points_x1d = positive_points[:,0]
    nmatch = int(ncoil * mul_ncoil_coeff)
    match_x1d = np.linspace(0.4, np.sqrt(2), nmatch)
    match_x1d[0] = 0.1
    # match_x1d = np.array([0.1, 0.5, 0.8, 1.2, 1.3, 1.414])
    # match_x1d = np.linspace(0.1, np.sqrt(2), nmatch)
    # match_x1d = get_uniform_points_on_ellipse(nmatch)
    # # match_x1d[0] = 0.1
    match_points = np.zeros((nmatch, 2))

    for i in range(nmatch):
        idx = np.argmin(np.abs(positive_points_x1d - match_x1d[i]))
        match_points[i,:] = positive_points[idx, :]
    

    if print_length_flag:
        print('\n'); print("*****"*5)
        print("number of positive sep points = %d" % positive_points.shape[0])
        print("number of matching points     = %d" % match_points.shape[0])
        print("matching points               :", match_points)
        print("*****"*5); print("\n")
    if match_points.shape[0] != np.unique(match_points, axis=0).shape[0]:
        raise ValueError("not enough match points so that there are duplications!")

    return match_points



def symmetric_coil_I_match_norm(xc1d, yc1d, match_points,\
                      x2d, y2d, psi2d_nan, p1, print_Ic1d_flag=False):
    """
    up-down symmetric coil match function
    :param xc1d: (1d np.ndarray) x coord of given coils
    :param yc1d: (1d np.ndarray) y coord of given coils
    :param match_points: (2d np.ndarray) coord of match points on separatrix
    :param x2d: (2d np.ndarray) normalized x2d
    :param y2d: (2d np.ndarray) normalized y2d
    :param psi2d_nan: (2d np.ndarray) normalized psi2d with np.nan in vacuum
    :param p1: (float) p1 in normailzed solovev GS equation
    :param print_Ic_flag: (bool) if True, print Ic array
    """
    upper_coil_num = xc1d.shape[0]
    # calculate j_\theta
    jth2d = x2d * p1
    matA = np.zeros((match_points.shape[0], upper_coil_num))
    vecb = np.zeros(match_points.shape[0])
    for i in range(matA.shape[0]):
        for j in range(matA.shape[1]):
            # ith sep_point, jth coil
            x_match = match_points[i,0]
            y_match = match_points[i,1]
            x_source = xc1d[j]
            y_source1 = yc1d[j]
            y_source2 = -yc1d[j]
            matA[i,j] = Green(x_match, y_match, x_source, y_source1) + \
                        Green(x_match, y_match, x_source, y_source2)
    for i in range(vecb.shape[0]):
        x_match = match_points[i,0]
        y_match = match_points[i,1]
        vecb[i] = -1.0 * intAreaGJ(x_match, y_match, x2d, y2d, jth2d, psi2d_nan)
    # return Ic1d
    Ic1d = np.linalg.lstsq(matA, vecb, rcond=None)[0]

    if print_Ic1d_flag:
        print('\n'); print("*****"*5)
        print("normalized upper coil currents are:", Ic1d)
        print("matA is :", matA)
        print("vecb is :", vecb)
        print("*****"*5); print("\n")
    return Ic1d


def calc_psi_tot_on_given_points(points2d, \
                                 xc1d_upper, yc1d_upper, Ic1d_upper,\
                                 x2d, y2d, psi2d_nan, p1, print_flag=False, plot_flag=False, fignum=444):
    jth2d = x2d * p1
    psi_on_points = np.zeros(points2d.shape[0])
    for i in range(psi_on_points.shape[0]):
        x = points2d[i, 0]
        y = points2d[i, 1]
        # calc coil contributions
        temp_coil_psi = 0.0
        if np.abs(x-0.0) < 1.e-14:
            psi_on_points[i] = 0.0
        else:
            for k in range(xc1d_upper.shape[0]):
                temp_coil_psi += (Green(x, y, xc1d_upper[k], yc1d_upper[k]) + \
                    Green(x, y, xc1d_upper[k], -yc1d_upper[k])) * Ic1d_upper[k]
            # calc FRC contributions
            temp_FRC_psi = intAreaGJ(x, y, x2d, y2d, jth2d, psi2d_nan)
            psi_on_points[i] = temp_coil_psi + temp_FRC_psi
    
    if print_flag:
        print("\n"); print("*****"*5)
        print("points are:", points2d)
        print("psi on given points are:", psi_on_points)
        print("*****"*5); print("\n")
    
    if plot_flag:
        temp_fig = plt.figure(num=fignum)
        temp_ax = temp_fig.add_subplot(111)
        temp_ax.scatter(np.arange(points2d.shape[0])+1, psi_on_points)
        temp_ax.set_title(r"$\psi$ on given points")
        temp_ax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    return psi_on_points














