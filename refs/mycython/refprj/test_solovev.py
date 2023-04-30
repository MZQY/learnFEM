import numpy as np
import matplotlib.pyplot as plt
from solovev import *
from check import check_GS, check_GS_af_match
from coil_match import *
from setvac import *


if __name__ == '__main__':
    rs = 0.2
    ro = rs / np.sqrt(2)
    xmin = 0.0; xmin_norm = xmin / ro
    xmax = 0.3; xmax_norm = xmax / ro
    ymin = -2.0; ymin_norm = ymin / ro
    ymax = 2.0; ymax_norm = ymax / ro
    zs = 1.0
    p1=-1.2
    Xp_x = 0.0
    Xp_y = zs / ro
    print("*****"*5)
    print("xmax_norm = %.2f" % xmax_norm)
    print("ymax_norm = %.2f" % ymax_norm)
    print("Xp_y norm = %.2f" % Xp_y)
    print("*****"*5)
    solovev = StaticSolovev(p1_norm=p1, Xpoint_x_norm=0.0, Xpoint_y_norm=Xp_y)
    solovev.print_coeffs()
    nx = 100
    ny = 200
    x2d, y2d, psi2d = solovev.get_psi_data(xmin_norm, \
                      xmax_norm, ymin_norm, ymax_norm, nx, ny)
    check_GS(p1, x2d, y2d, psi2d)
    solovev.plot_psi_contour(x2d, y2d, psi2d, fignum=1)

    # fig = plt.figure(2)
    # ax = fig.add_subplot(111)
    # ax.plot(y2d[0,:], psi2d[0,:])
    # print(psi2d[0,:])
    FRC_bool, sep_points = nan_psi_out_separatrix(Xp_y, x2d, y2d, psi2d)
    solovev.plot_psi_contour(x2d, y2d, FRC_bool, fignum=2)

    upper_ncoil=7

    psi2d_nan = np.where(FRC_bool, psi2d, np.nan)
    match_points = get_match_points(ncoil=upper_ncoil, sep_points=sep_points, mul_ncoil_coeff=3, print_length_flag=True)
    solovev.plot_psi_contour_with_points(x2d, y2d, psi2d_nan, match_points, fignum=4)

    xc1d_upper = np.ones(upper_ncoil)*xmax_norm*1.5
    yc1d_upper = np.linspace(Xp_y*0.1, Xp_y*1.0, upper_ncoil)
    # xc1d_upper = np.array([4.0, 4.0, 4.0, 4.0, 4.0])
    # yc1d_upper = np.array([1.0, 2.0, 5.0, 6.0, 7.0])
    Ic1d_upper = symmetric_coil_I_match_norm(xc1d=xc1d_upper, yc1d=yc1d_upper, match_points=match_points, \
        x2d=x2d, y2d=y2d, psi2d_nan=psi2d_nan, p1=p1, print_Ic1d_flag=True)

    xc1d = np.concatenate((np.flip(xc1d_upper), xc1d_upper))
    yc1d = np.concatenate((np.flip(-1.0 * yc1d_upper), yc1d_upper))
    Ic1d = np.concatenate((np.flip(Ic1d_upper), Ic1d_upper))
    print("******"*5)
    print("xc1d = ", xc1d)
    print("yc1d = ", yc1d)
    print("Ic1d = ", Ic1d)
    print("******"*5)

    calc_psi_tot_on_given_points(match_points, xc1d_upper, yc1d_upper, Ic1d_upper, x2d, y2d, psi2d_nan, p1, print_flag=True, plot_flag=True, fignum=444)
    psi_on_sep_points = calc_psi_tot_on_given_points(sep_points, xc1d_upper, yc1d_upper, Ic1d_upper, x2d, y2d, psi2d_nan, p1, print_flag=False, plot_flag=True, fignum=445)
    average_abs_psi_on_sep_points = np.average(np.abs(psi_on_sep_points))
    print("******"*5)
    print("average abs(psi) on separatrix: ", average_abs_psi_on_sep_points)
    print("******"*5)


    psi2d_vac = set_vacuum_psi(xc1d, yc1d, Ic1d, x2d, y2d, psi2d_nan, FRC_bool, p1)
    # print(np.asarray(psi2d_vac))
    solovev.plot_psi_contour(x2d, y2d, psi2d_vac, fignum=6)

    psi2d_tot = np.where(FRC_bool, psi2d_nan, psi2d_vac)
    solovev.plot_psi_contour(x2d, y2d, psi2d_tot, fignum=8)
    check_GS_af_match(p1, x2d, y2d, psi2d_tot,FRC_bool)
    solovev.plot_psi_3d(x2d, y2d, psi2d_tot, fignum=13)
    
    br2d, bz2d = solovev.get_Br_Bz_by_psi(x2d, y2d, psi2d_tot)
    solovev.plot_psi_contour(x2d, y2d, br2d, fignum=1111)
    solovev.plot_psi_contour(x2d, y2d, bz2d, fignum=1112)

    pres2d, jth2d = solovev.get_pres_jth2d(x2d, psi2d_tot, FRC_bool)
    solovev.plot_psi_contour(x2d, y2d, pres2d, fignum=2222)
    solovev.plot_psi_contour(x2d, y2d, jth2d, fignum=2223)

    plt.show()
