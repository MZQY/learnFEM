import basis
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
color_list = ['#3b6291', '#943c39', '#779043', '#624c7c', '#388498', '#bf7334', '#3f6899', '#9c403d', '#7d9847', '#675083', '#3b8ba1', '#c97937']


## test 2d basis functions
idx = 0
idy = 0
x2d, y2d = np.meshgrid(np.linspace(-0.5,1.5,100), np.linspace(-0.5,1.5,100),indexing='ij')
phi2d_01 = np.zeros(x2d.shape)
phi2d_02 = np.zeros(x2d.shape)
phi2d_03 = np.zeros(x2d.shape)

for i in range(x2d.shape[0]):
    for j in range(x2d.shape[1]):
        node_x = np.zeros(6)
        node_y = np.zeros(6)
        phi = np.zeros(6)
        basis.shape2d_t6(idx, idy, x2d[i,j], y2d[i,j], node_x, node_y, phi)
        phi2d_01[i,j] = phi[0]
        phi2d_02[i,j] = phi[1]
        phi2d_03[i,j] = phi[2]
        
fig4 = plt.figure(num=4)
ax4 = fig4.add_subplot(111, projection='3d')
ax4.plot_surface(x2d, y2d, phi2d_01, cmap='rainbow')
# ax4.plot_surface(x2d, y2d, phi2d_02, cmap='rainbow')
# ax4.plot_surface(x2d, y2d, phi2d_03, cmap='rainbow')
ax4.set_xlabel('x')
ax4.set_ylabel('y')
ax4.set_zlabel('z')
# ax4.scatter(node_x, node_y, np.zeros(node_x.shape[0]), c='b', lw=5)

plt.show()
