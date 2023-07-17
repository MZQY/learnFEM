"""
ymma, ymma98@qq.com
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

class PlotFig:
    """
    figure plot class.
    just a hodgepodge
    """
    def __init__(self):
        self.marker_list = ['o', 's', 'P', 'X',
                            'D', '^', '.', '<',
                            '>', 'v', '*', 'p',
                            '1', '2', '3', '4',
                            '+', 'x' ]
        self.line_style = ['-', '--', ':', '-.']
        # ref: https://sashamaps.net/docs/resources/20-colors/
        # with slight modification
        self.color_list = [
            '#e6194b', '#3cb44b',
            '#f58231', '#4363d8',
            '#911eb4', '#808080',
            '#46f0f0', '#f032e6',
            '#bcf60c', '#fabebe',
            '#008080', '#e6beff',
            '#9a6324', '#fffac8',
            '#800000', '#aaffc3',
            '#808000', '#ffd8b1',
            '#000075', '#ffe119',
            '#000000']
        # https://stackoverflow.com/questions/34859628/has-someone-made-the-parula-colormap-in-matplotlib
        # why parula: https://academia.stackexchange.com/questions/34327/clear-colormap-for-figures-when-printed
        from matplotlib.colors import LinearSegmentedColormap
        cm_data = [
            [0.2081, 0.1663, 0.5292],
            [0.2116238095, 0.1897809524, 0.5776761905],
            [0.212252381, 0.2137714286, 0.6269714286],
            [0.2081, 0.2386, 0.6770857143],
            [0.1959047619, 0.2644571429, 0.7279],
            [0.1707285714, 0.2919380952,0.779247619],
            [0.1252714286, 0.3242428571, 0.8302714286],
            [0.0591333333, 0.3598333333, 0.8683333333],
            [0.0116952381, 0.3875095238, 0.8819571429],
            [0.0059571429, 0.4086142857, 0.8828428571],
            [0.0165142857, 0.4266, 0.8786333333],
            [0.032852381, 0.4430428571, 0.8719571429],
            [0.0498142857, 0.4585714286, 0.8640571429],
            [0.0629333333, 0.4736904762, 0.8554380952],
            [0.0722666667, 0.4886666667,0.8467],
            [0.0779428571, 0.5039857143, 0.8383714286],
            [0.079347619, 0.5200238095, 0.8311809524],
            [0.0749428571, 0.5375428571, 0.8262714286],
            [0.0640571429, 0.5569857143, 0.8239571429],
            [0.0487714286, 0.5772238095, 0.8228285714],
            [0.0343428571, 0.5965809524, 0.819852381],
            [0.0265, 0.6137, 0.8135],
            [0.0238904762, 0.6286619048, 0.8037619048],
            [0.0230904762, 0.6417857143, 0.7912666667],
            [0.0227714286, 0.6534857143, 0.7767571429],
            [0.0266619048, 0.6641952381, 0.7607190476],
            [0.0383714286, 0.6742714286, 0.743552381],
            [0.0589714286, 0.6837571429, 0.7253857143],
            [0.0843, 0.6928333333, 0.7061666667],
            [0.1132952381, 0.7015, 0.6858571429],
            [0.1452714286, 0.7097571429, 0.6646285714],
            [0.1801333333, 0.7176571429,0.6424333333],
            [0.2178285714, 0.7250428571, 0.6192619048],
            [0.2586428571, 0.7317142857, 0.5954285714],
            [0.3021714286, 0.7376047619, 0.5711857143],
            [0.3481666667, 0.7424333333, 0.5472666667],
            [0.3952571429, 0.7459, 0.5244428571],
            [0.4420095238, 0.7480809524, 0.5033142857],
            [0.4871238095, 0.7490619048, 0.4839761905],
            [0.5300285714, 0.7491142857, 0.4661142857],
            [0.5708571429, 0.7485190476, 0.4493904762],
            [0.609852381, 0.7473142857, 0.4336857143],
            [0.6473, 0.7456, 0.4188],
            [0.6834190476, 0.7434761905, 0.4044333333],
            [0.7184095238, 0.7411333333, 0.3904761905],
            [0.7524857143, 0.7384, 0.3768142857],
            [0.7858428571, 0.7355666667, 0.3632714286],
            [0.8185047619, 0.7327333333, 0.3497904762],
            [0.8506571429, 0.7299, 0.3360285714],
            [0.8824333333, 0.7274333333, 0.3217],
            [0.9139333333, 0.7257857143, 0.3062761905],
            [0.9449571429, 0.7261142857, 0.2886428571],
            [0.9738952381, 0.7313952381, 0.266647619],
            [0.9937714286, 0.7454571429, 0.240347619],
            [0.9990428571, 0.7653142857, 0.2164142857],
            [0.9955333333, 0.7860571429, 0.196652381],
            [0.988, 0.8066, 0.1793666667],
            [0.9788571429, 0.8271428571, 0.1633142857],
            [0.9697, 0.8481380952, 0.147452381],
            [0.9625857143, 0.8705142857, 0.1309],
            [0.9588714286, 0.8949, 0.1132428571],
            [0.9598238095, 0.9218333333,0.0948380952],
            [0.9661, 0.9514428571, 0.0755333333],
            [0.9763, 0.9831, 0.0538]]
        self.parula_map =\
            LinearSegmentedColormap.from_list('parula', cm_data)

    def add_conf_ax(self,\
                   fig, nrows, ncols, idx,
                   x2d, y2d, z2d,
                   title = "",
                   xlabel="",
                   ylabel="",
                   lines=20,
                   cbar_formatter_scale=-1,
                   num_in_cbar = 7,
                   aspect = None,
                   color_map = "",
                   x_log_scale = False,
                   y_log_scale = False
                   ):
        """
        add contour-fill plot to existing figure.
        :param fig: plt.figure() object
        :param nrows: (int) total row number of figure
        :param ncols: (int) total col number of figure
        :param idx: (int) index of current ax in figure
        :param x2d: (np.ndarray)
        :param y2d: (np.ndarray)
        :param z2d: (np.ndarray)
        :param title: (str) title of axes
        :param xlabel: (str) x-label of axes
        :param ylabel: (str) y-label of axes
        :param lines: (int) contour line number
        :param cbar_formatter_scale: (int) scale of cbar, 1eX
               if cbar_formatter_scale = -1, the color bar
               will not be plotted
        :param num_in_cbar: (int) number shown in color bar
        :param aspect: (float) larger aspect, wider the x-axis
                default: the force aspect is not used
        :param color_map: (str) the default colormap is parula_map,
                which is good enough
        :param x_log_scale: (bool) if True, scale of x-axis will be log
        :param y_log_scale: (bool) if True, scale of y-axis will be log
        """
        ax = fig.add_subplot(nrows, ncols, idx)
        ax.ticklabel_format(style='sci', scilimits=(-2,2))
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if aspect:
            _plt_forceAspect(ax, aspect=aspect)
        if x_log_scale:
            ax.set_xscale('log')
        if y_log_scale:
            ax.set_yscale('log')
        if color_map:
            cmap = color_map
        else:
            cmap = self.parula_map
        cs = ax.contourf(x2d, y2d, z2d, lines, cmap=cmap)
        # remove white lines of contour: https://stackoverflow.com/questions/8263769/hide-contour-linestroke-on-pyplot-contourf-to-get-only-fills
        for c in cs.collections:
            c.set_edgecolor("face")

        # plot color bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if abs(cbar_formatter_scale-(-1))>1.e-14:
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes("right",
                                                size="3%",
                                                pad=0.2)
            cbar = fig.colorbar(cs, cax=colorbar_axes,
                    format=OOMFormatter(cbar_formatter_scale,
                                        mathText=False))
            # show n numbers in cbar
            cbar.ax.locator_params(nbins=num_in_cbar)
        else:
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes("right",
                                                size="3%",
                                                pad=0.2)
            cbar = fig.colorbar(cs, cax=colorbar_axes, format=ticker.FuncFormatter(_fmt))
            # show n numbers in cbar
            cbar.ax.locator_params(nbins=num_in_cbar)


    def add_con_ax(self,\
                   fig, nrows, ncols, idx,
                   x2d, y2d, z2d,
                   title = "",
                   xlabel="",
                   ylabel="",
                   lines=20,
                   cbar_formatter_scale=-1,
                   num_in_cbar = 7,
                   aspect = None,
                   color_map = "",
                   x_log_scale = False,
                   y_log_scale = False
                   ):
        """
        add contour plot to existing figure.
        :param fig: plt.figure() object
        :param nrows: (int) total row number of figure
        :param ncols: (int) total col number of figure
        :param idx: (int) index of current ax in figure
        :param x2d: (np.ndarray)
        :param y2d: (np.ndarray)
        :param z2d: (np.ndarray)
        :param title: (str) title of axes
        :param xlabel: (str) x-label of axes
        :param ylabel: (str) y-label of axes
        :param lines: (int) contour line number
        :param cbar_formatter_scale: (int) scale of cbar, 1eX
               if cbar_formatter_scale = -1, the color bar
               will not be plotted
        :param num_in_cbar: (int) number shown in color bar
        :param aspect: (float) larger aspect, wider the x-axis
                default: the force aspect is not used
        :param color_map: (str) the default colormap is parula_map,
                which is good enough
        :param x_log_scale: (bool) if True, scale of x-axis will be log
        :param y_log_scale: (bool) if True, scale of y-axis will be log
        """
        ax = fig.add_subplot(nrows, ncols, idx)
        ax.ticklabel_format(style='sci', scilimits=(-2,2))
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if aspect:
            _plt_forceAspect(ax, aspect=aspect)
        if x_log_scale:
            ax.set_xscale('log')
        if y_log_scale:
            ax.set_yscale('log')
        if color_map:
            cmap = color_map
        else:
            cmap = self.parula_map
        cs = ax.contour(x2d, y2d, z2d, lines, cmap=cmap)

        # plot color bar
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        if abs(cbar_formatter_scale-(-1))>1.e-14:
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes("right",
                                                size="3%",
                                                pad=0.2)
            cbar = fig.colorbar(cs, cax=colorbar_axes,
                    format=OOMFormatter(cbar_formatter_scale,
                                        mathText=False))
            # show n numbers in cbar
            cbar.ax.locator_params(nbins=num_in_cbar)
        else:
            divider = make_axes_locatable(ax)
            colorbar_axes = divider.append_axes("right",
                                                size="3%",
                                                pad=0.2)
            cbar = fig.colorbar(cs, cax=colorbar_axes, format=ticker.FuncFormatter(_fmt))
            # show n numbers in cbar
            cbar.ax.locator_params(nbins=num_in_cbar)



    def add_1d_ax(self,
                  fig, nrows, ncols, idx,
                  x1d, y1d,
                  title ="",
                  xlabel="",
                  ylabel="",
                  color="b",
                  line_style="-",
                  label="",
                  x_log_scale = False,
                  y_log_scale = False,
                  scatter_plot=False,
                  scatter_marker='s'
                 ):
        """
        add contour plot to existing figure.
        :param fig: plt.figure() object
        :param nrows: (int) total row number of figure
        :param ncols: (int) total col number of figure
        :param idx: (int) index of current ax in figure
        :param x1d: (np.ndarray)
        :param y1d: (np.ndarray)
        :param title: (str) title of axes
        :param xlabel: (str) x-label of axes
        :param ylabel: (str) y-label of axes
        :param color: (str) line color, default: blue
        :param line_style: (str) line style of the line plot
                         default: solid line
        :param label: (str) label of the curve
        :param x_log_scale: (bool) if True, scale of x-axis will be log
        :param y_log_scale: (bool) if True, scale of y-axis will be log
        :param scatter_plot: (bool) if True, ax will be plt.scatter()
        :param scatter_marker: (str) if scatter_plot=True, then marker
                            style will be scatter_marker
        """
        ax = fig.add_subplot(nrows, ncols, idx)
        ax.ticklabel_format(style='sci', scilimits=(-2,2))
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        if x_log_scale:
            ax.set_xscale('log')
        if y_log_scale:
            ax.set_yscale('log')
        ax.grid(linestyle='-.')
        if scatter_plot:
            ax.scatter(x1d, y1d,
                       edgecolors=color,
                       facecolors='none',
                       marker=scatter_marker, label=label)
        else:
            ax.plot(x1d, y1d, color=color,
                    linestyle=line_style,
                    label = label,
                    )
        if label:
            ax.legend()


class OOMFormatter(ticker.ScalarFormatter):
    # https://newbedev.com/
    # python-matplotlib-colorbar-scientific-notation-base
    def __init__(self, order=0, fformat="%1.2f",
                 offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self,
                                        useOffset=offset,
                                        useMathText=mathText)

    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom

    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % ticker._mathdefault(self.format)


def _plt_forceAspect(ax,aspect=1):
    # aspect is width/height
    # https://stackoverflow.com/questions/7965743/how-can-i-set-the-aspect-ratio-in-matplotlib
    import scipy
    scale_str = ax.get_yaxis().get_scale()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if scale_str == 'linear':
        asp = abs((xmax - xmin) / (ymax - ymin)) / aspect
    elif scale_str == 'log':
        asp = abs((scipy.log(xmax) - scipy.log(xmin)) / (scipy.log(ymax) - scipy.log(ymin))) / aspect
    ax.set_aspect(asp)

def _fmt(x, pos):
    a, b = '{:.2e}'.format(x).split('e')
    b = int(b)
    return r'${} \times 10^{{{}}}$'.format(a, b)

