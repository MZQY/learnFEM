
import numpy as np
import matplotlib.pyplot as plt
import findiff
# from check import check_GS
# from coil_match import *

class StaticSolovev:
    def __init__(self, p1_norm=1.0,
                 Xpoint_x_norm = 0.0,
                 Xpoint_y_norm = 0.0,
                 ):
        """
        initialize the Solovev solver, x and y are normalized by r_o
        we have 5 BC constraints:
        1. rs point curvature
        2. high point (upper X point) curvature
        3. O point dpsi/dx = 0
        4. psi = 0 at X point
        5. psi = 0 at rs point
        :param p1_norm: (float) p1 = dp/dpsi, where the p and psi are normalized
        :param Xpoint_x_norm: (float) x of X point's coord (x, y), which is normalized
        :param Xpoint_y_norm: (float) y of X point's coord (x, y), which is normalized
        """
        # define normalization parameters
        self.p1 = p1_norm
        self.Xp_x = Xpoint_x_norm
        self.Xp_y = Xpoint_y_norm
        # some constants
        self.mu0 = 4 * np.pi * 1.e-7
        self.mi = 1.67262192e-27
        self.e = 1.60217663e-19
        # coeffs for solution
        self.c = np.zeros(5)
        self.calc_coeff_by_BC()

    def psi(self, x, y, ndx=0, ndy=0):
        res =  self.c[0] * self.psi1(x, y, ndx, ndy) \
             + self.c[1] * self.psi2(x, y, ndx, ndy) \
             + self.c[2] * self.psi3(x, y, ndx, ndy) \
             + self.c[3] * self.psi4(x, y, ndx, ndy) \
             + self.c[4] * self.psi5(x, y, ndx, ndy) \
             + self.psip(x, y, ndx, ndy)
        return res



    def calc_coeff_by_BC(self):
        """
        calculate c1 to c5 by boundary conditions.
        1. dpsi/dx = 0 (X point)
        2. dpsi/dy = 0 (X point)
        3. dpsi/dx = 0 (O point)
        4. psi = 0     (X point)
        5. psi = 0     (r_s point)
        """
        kappa = self.Xp_y
        # N1 = -2.0/kappa**2
        # N3 = -kappa/4.0
        N1 = -np.sqrt(2)/kappa**2
        N3 = -kappa/2.0
        A_mat = np.array([
            [self.psi1(np.sqrt(2), 0.0, ndy=2)+N1*self.psi1(np.sqrt(2), 0.0, ndx=1),
             self.psi2(np.sqrt(2), 0.0, ndy=2)+N1*self.psi2(np.sqrt(2), 0.0, ndx=1),
             self.psi3(np.sqrt(2), 0.0, ndy=2)+N1*self.psi3(np.sqrt(2), 0.0, ndx=1),
             self.psi4(np.sqrt(2), 0.0, ndy=2)+N1*self.psi4(np.sqrt(2), 0.0, ndx=1),
             self.psi5(np.sqrt(2), 0.0, ndy=2)+N1*self.psi5(np.sqrt(2), 0.0, ndx=1),
            ],
            [self.psi1(0, kappa, ndx=2)+N3*self.psi1(0, kappa, ndy=1),
             self.psi2(0, kappa, ndx=2)+N3*self.psi2(0, kappa, ndy=1),
             self.psi3(0, kappa, ndx=2)+N3*self.psi3(0, kappa, ndy=1),
             self.psi4(0, kappa, ndx=2)+N3*self.psi4(0, kappa, ndy=1),
             self.psi5(0, kappa, ndx=2)+N3*self.psi5(0, kappa, ndy=1),
            ],
            [self.psi1(1.0, 0.0, ndx=1), self.psi2(1.0, 0.0, ndx=1),
             self.psi3(1.0, 0.0, ndx=1), self.psi4(1.0, 0.0, ndx=1),
             self.psi5(1.0, 0.0, ndx=1)],
            [self.psi1(self.Xp_x, self.Xp_y), self.psi2(self.Xp_x, self.Xp_y),
             self.psi3(self.Xp_x, self.Xp_y), self.psi4(self.Xp_x, self.Xp_y),
             self.psi5(self.Xp_x, self.Xp_y)],
            [self.psi1(np.sqrt(2), 0.0), self.psi2(np.sqrt(2), 0.0),
             self.psi3(np.sqrt(2), 0.0), self.psi4(np.sqrt(2), 0.0),
             self.psi5(np.sqrt(2), 0.0)]
        ])
        B_mat = -1.0 * np.array([
            self.psip(np.sqrt(2), 0.0, ndy=2)+N1*self.psip(np.sqrt(2), 0.0, ndx=1),
            self.psip(0, kappa, ndx=2)+N3*self.psip(0, kappa, ndy=1),
            self.psip(1.0, 0.0, ndx=1),
            self.psip(self.Xp_x, self.Xp_y),
            self.psip(np.sqrt(2), 0.0)
        ])
        self.c = np.linalg.solve(A_mat, B_mat)


    def psi1(self, x, y, ndx=0, ndy=0):
        """
        first homogeneous solution, psi1 = 1
        """
        if not (isinstance(ndx,int) and isinstance(ndy,int) \
                and (ndx>=0) and (ndy>=0)):
            raise TypeError("ndx and ndy must be 0 or positive integers!")
        if (ndx>=3 or ndy>=3):
            raise ValueError("ndx and ndy should always <=2 in GS equation!")
        elif (ndx != 0 and ndy != 0):
            raise ValueError("ndx and ndy can not >0 at the same time!")

        if (ndx==0 and ndy==0):
            return 1.0
        else:
            return 0.0

    def psi2(self, x, y, ndx=0, ndy=0):
        """
        second homogeneous solution, psi2 = x^2
        """
        if not (isinstance(ndx,int) and isinstance(ndy,int) \
                and (ndx>=0) and (ndy>=0)):
            raise TypeError("ndx and ndy must be 0 or positive integers!")
        if (ndx>=3 or ndy>=3):
            raise ValueError("ndx and ndy should always <=2 in GS equation!")
        elif (ndx != 0 and ndy != 0):
            raise ValueError("ndx and ndy can not >0 at the same time!")

        if (ndy == 0):
            if (ndx==0):
                return x**2
            elif (ndx==1):
                return 2.0*x
            elif (ndx==2):
                return 2.0
            else:
                return 0.0
        else:
            return 0.0

    def psi3(self, x, y, ndx=0, ndy=0):
        """
        third homogeneous solution, psi4 = x^4 - 4 * x^2 * y^2
        """
        if not (isinstance(ndx,int) and isinstance(ndy,int) \
                and (ndx>=0) and (ndy>=0)):
            raise TypeError("ndx and ndy must be 0 or positive integers!")
        if (ndx>=3 or ndy>=3):
            raise ValueError("ndx and ndy should always <=2 in GS equation!")
        elif (ndx != 0 and ndy != 0):
            raise ValueError("ndx and ndy can not >0 at the same time!")

        if (ndx==0 and ndy==0):
            return x**4 - 4*x**2*y**2
        elif (ndx==1 and ndy==0):
            return 4*x**3 - 8*x*y**2
        elif (ndx==2 and ndy==0):
            return 4.0*(3.0*x**2 - 2.0*y**2)
        elif (ndx==0 and ndy==1):
            return -8.0*x**2*y
        elif (ndx==0 and ndy==2):
            return -8.0*x**2
        else:
            raise ValueError("wrong ndx and ndy")


    def psi4(self, x, y, ndx=0, ndy=0):
        """
        fourth homogeneous solution, psi4 = x^6 - 12 x^4 y^2 + 8 x^2 y^4
        """
        if not (isinstance(ndx,int) and isinstance(ndy,int) \
                and (ndx>=0) and (ndy>=0)):
            raise TypeError("ndx and ndy must be 0 or positive integers!")
        if (ndx>=3 or ndy>=3):
            raise ValueError("ndx and ndy should always <=2 in GS equation!")
        elif (ndx != 0 and ndy != 0):
            raise ValueError("ndx and ndy can not >0 at the same time!")

        if (ndx==0 and ndy==0):
            return x**6 - 12*x**4*y**2 + 8*x**2*y**4
        elif (ndx==1 and ndy==0):
            return 6*x**5 - 48*x**3*y**2 + 16*x*y**4
        elif (ndx==2 and ndy==0):
            return 30*x**4 - 144*x**2*y**2 + 16*y**4
        elif (ndx==0 and ndy==1):
            return -24*x**4*y + 32*x**2*y**3
        elif (ndx==0 and ndy==2):
            return 24*x**2*(-x**2 + 4*y**2)
        else:
            raise ValueError("wrong ndx and ndy")



    def psi5(self, x, y, ndx=0, ndy=0):
        """
        fifth homogeneous solution, psi5 = x^8 - 24 x^6 y^2 + 48 x^4 y^4 - 64/5 x^2 y^6
        """
        if not (isinstance(ndx,int) and isinstance(ndy,int) \
                and (ndx>=0) and (ndy>=0)):
            raise TypeError("ndx and ndy must be 0 or positive integers!")
        if (ndx>=3 or ndy>=3):
            raise ValueError("ndx and ndy should always <=2 in GS equation!")
        elif (ndx != 0 and ndy != 0):
            raise ValueError("ndx and ndy can not >0 at the same time!")

        if (ndx==0 and ndy==0):
            return x**8 - 24*x**6*y**2 + 48*x**4*y**4 - 64*x**2*y**6/5
        elif (ndx==1 and ndy==0):
            return 8*x*(x**6 - 18*x**4*y**2 + 24*x**2*y**4 - 16*y**6/5)
        elif (ndx==2 and ndy==0):
            return 56*x**6 - 720*x**4*y**2 + 576*x**2*y**4 - 128*y**6/5
        elif (ndx==0 and ndy==1):
            return 48*x**2*y*(-x**4 + 4*x**2*y**2 - 8*y**4/5)
        elif (ndx==0 and ndy==2):
            return 48*x**2*(-x**4 + 12*x**2*y**2 - 8*y**4)
        else:
            raise ValueError("wrong ndx and ndy")


    def psip(self, x, y, ndx=0, ndy=0):
        """
        particular solution, psip = - p1/8 x^4
        """
        if not (isinstance(ndx,int) and isinstance(ndy,int) \
                and (ndx>=0) and (ndy>=0)):
            raise TypeError("ndx and ndy must be 0 or positive integers!")
        if (ndx>=3 or ndy>=3):
            raise ValueError("ndx and ndy should always <=2 in GS equation!")
        elif (ndx != 0 and ndy != 0):
            raise ValueError("ndx and ndy can not >0 at the same time!")

        if (ndx==0 and ndy==0):
            return -self.p1 / 8 * x**4
        elif (ndx==1 and ndy==0):
            return  -self.p1 / 2 * x**3
        elif (ndx==2 and ndy==0):
            return -self.p1 * 1.5 * x**2
        elif (ndx==0 and ndy==1):
            return 0.0
        elif (ndx==0 and ndy==2):
            return 0.0
        else:
            raise ValueError("wrong ndx and ndy")


    def print_coeffs(self):
        print("*****"*5)
        print("c_mat = {}".format(self.c))
        print("*****"*5)

    def get_psi_data(self,
                    xmin_norm, xmax_norm,\
                    ymin_norm, ymax_norm, nx, ny \
                    ):
        x1d = np.linspace(xmin_norm, xmax_norm, nx)
        y1d = np.linspace(ymin_norm, ymax_norm, ny)
        x2d, y2d = np.meshgrid(x1d, y1d, indexing='ij')
        psi2d = self.psi(x2d, y2d)
        return x2d, y2d, psi2d
    
    
    def get_Br_Bz_by_psi(self, x2d_norm, y2d_norm, psi2d_norm):
        dx = x2d_norm[1,0] - x2d_norm[0,0]
        dy = y2d_norm[0,1] - y2d_norm[0,0]
        dr = findiff.FinDiff(0, dx, 1, acc=4)
        dz = findiff.FinDiff(1, dy, 1, acc=4)
        dpsidr = dr(psi2d_norm)
        dpsidz = dz(psi2d_norm)
        print("AAAAAAAAAA")
        print(dpsidr.shape)
        print(x2d_norm.shape)
        print(dpsidz[0,0])
        br2d = np.where(np.abs(x2d_norm)<1.e-14, -dpsidz / dx,  -dpsidz/x2d_norm)
        bz2d = np.where(np.abs(x2d_norm)<1.e-14, dpsidr / dx,   dpsidr/x2d_norm)
        return br2d, bz2d
    
    def get_pres_jth2d(self, x2d_norm, psi2d_norm, FRC_bool):
        pres2d = np.where(FRC_bool, self.p1 * psi2d_norm, 0.0)
        jth2d = np.where(FRC_bool, x2d_norm * self.p1, 0.0)
        return pres2d, jth2d


    def plot_psi_contour(self, x2d, y2d, psi2d, fignum):
        temp_fig = plt.figure(num=fignum)
        temp_ax = temp_fig.add_subplot(121)
        temp_cs = temp_ax.contour(x2d, y2d, psi2d,\
                        levels=50, cmap='rainbow')
        temp_ax.contour(x2d, y2d, psi2d, [0.0], colors='k')
        temp_ax2 = temp_fig.add_subplot(122)
        temp_ax2.plot(x2d[:, int(x2d.shape[1]/2)], \
                      psi2d[:, int(x2d.shape[1]/2)])
        temp_fig.colorbar(temp_cs)


    def plot_psi_contour_with_points(self, x2d, y2d, psi2d, points2d, fignum):
        temp_fig = plt.figure(num=fignum)
        temp_ax = temp_fig.add_subplot(121)
        temp_cs = temp_ax.contour(x2d, y2d, psi2d,\
                        levels=50, cmap='rainbow')
        temp_ax.contour(x2d, y2d, psi2d, [0.0], colors='k')
        temp_ax.scatter(points2d[:,0], points2d[:,1])
        temp_ax2 = temp_fig.add_subplot(122)
        temp_ax2.plot(x2d[:, int(x2d.shape[1]/2)], \
                      psi2d[:, int(x2d.shape[1]/2)])
        temp_fig.colorbar(temp_cs)


    def plot_psi_3d(self, x2d, y2d, psi2d, fignum):
        temp_fig = plt.figure(num=fignum)
        temp_ax = temp_fig.add_subplot(111, projection='3d')
        temp_cs = temp_ax.plot_surface(x2d, y2d, psi2d, cmap='rainbow')
        temp_fig.colorbar(temp_cs)




