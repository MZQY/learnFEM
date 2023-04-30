import numpy as np
import matplotlib.pyplot as plt


def err_plot_for_nx200(fignum):
    tempfig = plt.figure(num=fignum)
    tempax = tempfig.add_subplot(111)
    abspsi_on_sep = np.array([0.0055072344207646225,  # 50
                              0.0017425544959872878,  # 100
                              0.0009510643906819591,  # 150
                              0.0006835013387137803,  # 200
                              0.000452749066601147,   # 250
                              0.0003845303973921806,  # 300
                              0.000344289878606281,   # 350
                              0.0002840770725707699,  # 400
                              0.00038800477946517167, # 450, weird
                              0.00025204302033117617, # 500
                              0.0002511622650832245,  # 550
                              0.00023533483988733993, # 600
                              0.00025687714506898154, # 650
                              0.0002499615196005229,  # 700
                              ])
    ny = np.array([50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700])
    tempax.scatter(ny, abspsi_on_sep, marker='o', c='r')
    tempax.plot(ny, abspsi_on_sep, color='b')
    tempax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    tempax.set_title(r"average $|\psi|$ on separatrix with nx=200")
    tempax.set_xlabel("ny")
    tempax.set_ylabel(r"average $|\psi|$")




def err_plot_for_ny200(fignum):
    tempfig = plt.figure(num=fignum)
    tempax = tempfig.add_subplot(111)
    abspsi_on_sep = np.array([0.003797482870011187,  # 20
                              0.001376627174526312,  # 50
                              0.0009468790554719427, # 100
                              0.0006808921415995716, # 150
                              0.0006835013387137803, # 200
                              0.0006589089526600003, # 250
                              0.0005289046519532492, # 300
                              0.0004886258435643118, # 350
                              0.0005979913695077594, # 400
                              0.0004989622044892994, # 450
                              0.0004603191439382358, # 500
                              ])
    nx = np.array([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500])
    tempax.scatter(nx, abspsi_on_sep, marker='o', c='r')
    tempax.plot(nx, abspsi_on_sep, color='b')
    tempax.ticklabel_format(axis='y', style='sci', scilimits=(-1,1))
    tempax.set_title(r"average $|\psi|$ on separatrix with ny=200")
    tempax.set_xlabel("nx")
    tempax.set_ylabel(r"average $|\psi|$")



if __name__=='__main__':
    err_plot_for_nx200(1)
    err_plot_for_ny200(2)
    plt.show()
    pass