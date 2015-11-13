#!usr/local/env python
# use '#!usr/local/bin/anaconda/bin/python' for specific path execution
# try '#!usr/local/bin python' also#
#
# Functional script for plotting figures in other scripts.
# Usage:
#   import *_plotting as *plt  where * is the name of the master script
#   For example, if the master script is tresp.py, then this script should
#   be renamed as tresp_plotting.py, and the usage will be:
#   import tresp_plotting as trespplt
#
# Revision history:
# 20150915 -added golden ratio for figsize aspect ratio, jbp3
#          -added fontsize adjustment for tick labels, jbp3
#          -added equal axis plotting (square axes), jbp3
# 20151111 -added positional and named arguments, jbp3
#          -changed to standalone script for testing, jbp3
#

# import library modules
import numpy as np
import matplotlib.pyplot as plt


def basicplot(data, param_dict, xlabel="x-data", ylabel="y-data",
              suptitle="super title", title="title", xmin=0.,
              xmax=10., squareaxes=False, filename='test.png'):
    """
    A helper function to make a graph.
    Derived from: http://matplotlib.org/faq/usage_faq.html
    Example:
    basicplot((x, y),
              {'color': 'b', 'linestyle': 'dashed', 'label': 'data'},
              xlabel="test")
    Parameters
    ----------
    data : tuple of arrays
        Example: data=(x, (y1, y2))
        The x, y data to plot. This is a tuple of either one
        or two arrays.

    param_dict : tuple of dict
        Dictionary of kwargs to pass to plt.plot

    **kwargs :
        Keyword arguments for various parts of the plot.
    Returns
    -------
 
    """
    # range of linestyles and linewidths
    ls = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-',
          '--', '-.', ':', '-', '--', '-.', ':', '--', '-.',
          ':', '--', '-.', ':', '-', '--', '-.', ':', '-',
          '--', '-.', ':']
    lw = np.array([1., 1., 1., 1., 1.5, 1.5, 1.5, 1.5,
                   2., 2., 2., 2., 2.5, 2.5, 2.5, 2.5,
                   3., 3., 3., 3., 3.5, 3.5, 3.5, 3.5,
                   4., 4., 4., 4., 4.5, 4.5, 4.5, 4.5])
    
    # initialize instance of figure
    plt.style.use('bakerplotstyle')
    goldratio = 1.618  # golden ratio for aesthetically pleasing plot
    plotwidth = 3.25  # inches
    fig = plt.figure(dpi=150, facecolor='0.8',
                     figsize=(plotwidth, plotwidth/goldratio))
    ax = plt.subplot(111)
    for i, y in enumerate(data[1]):
        plt.plot(data[0], y, linestyle=ls[i], linewidth=lw[i],
                 **param_dict[i])
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.suptitle(suptitle, fontsize=12, y=1.0)
    plt.title(title, fontsize=10)
    # plt.text(3.5, .9, 'centered title',
    #         horizontalalignment='center')

    plt.xlim(xmin, xmax)
    if squareaxes is True:
        plt.gca().set_aspect('equal', adjustable='box')
    else:
        # plt.ylim(ymin, ymax)
        pass
    # plt.grid() # this is set in the bakerplotstyle sheet
    box = ax.get_position()
    ax.set_position([box.x0+box.width*0.05, box.y0+box.height*0.05,
                     box.width*0.85, box.height*0.85])
    plt.legend(loc='center left',
               bbox_to_anchor=(1.05, 0.50),
               borderaxespad=0., labelspacing=0.1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=10)
    # plt.legend(loc='upper right')
    # change font size of tick labels
    plt.tick_params(axis='both', which='major', labelsize=8)
    plt.tick_params(axis='both', which='minor', labelsize=6)

    plt.savefig(filename, dpi=150, bbox_inches='tight')

    # display figure
    plt.show()
    # clear from memory
    plt.close(fig)


if __name__ == "__main__":

    d1 = np.linspace(0., 2.*np.pi, num=600)
    d2 = np.cos(2.*np.pi*1*d1)
    xlabel = r'$\omega={:4.1f}$'.format(14.3)
    basicplot([d1,  d2],
              {'color': 'b', 'linestyle': 'dashed', 'label': 'data'},
              xlabel=xlabel)
