#!usr/local/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import ode
from scipy import interpolate
import cmath
# from scipy.signal import lti, step, impulse
import scipy.signal as scisig
import smd_plotting as smdplt
import rungekutta as rk

# thomson example 4.8.2 parameters
# k = 100.
# b = 8.0
# m = 2.0

# luke's parameters
k = 1.3
b = 0.8
m = 1.3


def smd(t, x):

    xp = np.zeros((2, 1), dtype=float)
    # print t
    xp[0] = x[1]
    # forceinterp = interpolate.interp1d(time, forcing, kind='nearest',
    #                                   bounds_error=False, fill_value=0.)
    # print t
    # f = forceinterp(t)
    f = forcing_helper(t)
    # f = forcing_thomson(t)
    xp[1] = f/m - b/m*x[1] - k/m*x[0]

    return xp


def forcing_helper(t):

    for i, tau in enumerate(time):
        if (t >= 30. and t <= 31.) or (t >= 35. and t <= 36.):
        # if (t >= 30. and t <= 60.):
            f = 1.
        else:
            f = 0.

    return f


def forcing_thomson(t):

    if t < 0.25:
        f = 4.*t
    elif t < 0.5:
        f = -2.*(t-0.25) + 1.0
    elif t < 1.0:
        f = -1.*(t-0.5) + 0.5
    else:
        f = 0.

    return f


def laplacesolution(t, a):

    d = cmath.sqrt(b**2 - 4*k*m)
    if t < a:
        y = 0.
    elif t >= a:
        y = -1/d*np.exp(-(b+d)/(2*m)*(t-a)) + 1/d*np.exp(-(b-d)/(2*m)*(t-a))
    else:
        y = 0.

    return y


def transfunc():

    num = [1]
    den = [m, b, k]

    tf = scisig.lti(num, den)

    return tf


def ltisys(T, U):

    tf = transfunc()
    t, yout, xout = scisig.lsim(tf, U=U, T=T)
    # t, s = scisig.step(tf, T=T)
    # t, i = scisig.impulse(tf, T)

    # data = (t, (yout, U, xout[:, 0], xout[:, 1]))
    # params = ({'label': 'response'},
    #          {'label': 'input'},
    #          {'label': 'xout[:, 0]'},
    #          {'label': 'xout[:, 1]'})
    # smdplt.basicplot(data, params, xlabel='time, [s]', ylabel='response',
    #                 suptitle=' ', title='Spring-Mass-Damper Time Response',
    #                 xmax=np.max(T), filename='smd_response_ltisys.png')

    return yout


def impresp():

    tf = transfunc()
    t, yout = scisig.impulse(tf)

    return yout

if __name__ == "__main__":

    time = np.linspace(0., 100., num=1000)
    forcing = np.zeros_like(time)
    for i, tau in enumerate(time):
        forcing[i] = forcing_helper(tau)  # _thomson(tau)
    
    # lti system forcing using scisig.lsim
    lsimresponse = ltisys(time, forcing)

    # impulse response using convolution (Luke's method)
    datair = np.loadtxt('smd_ir.txt', delimiter=',', dtype=float)

    # response calc using runge-kutta integration method
    # datark = np.loadtxt('smd_rk.txt', delimiter=',', dtype=float)
    xinit = np.array(([0., 0.]), dtype=float)
    deltat = 0.01
    tzero = np.min(time)
    tfinal = np.max(time)
    trk, solrk = rk.runga(smd, xinit, tzero, tfinal, deltat)
    finterp = interpolate.interp1d(trk, solrk[:, 0], bounds_error=False,
                                   fill_value=0., kind='linear')
    solrknew = finterp(time)

    # scipy.signal.convolve
    impresponse = impresp()
    convresp = scisig.convolve(forcing, impresponse,
                               mode='full')  # /np.sum(impresponse)

    datasets = (time,
                (forcing, datair[:, 1], lsimresponse, solrknew, convresp[0:1000]))
    params = ({'label': 'forcing'},
              {'label': 'conv. response'},
              {'label': 'lsim response'},
              {'label': 'runge-kutta response'},
              {'label': 'scipy.convolve response'})
    smdplt.basicplot(datasets, params, xlabel='time, [s]', ylabel='response',
                     suptitle=' ', title='Spring-Mass-Damper Time Response',
                     xmax=np.max(time), filename='smd_response.png')
