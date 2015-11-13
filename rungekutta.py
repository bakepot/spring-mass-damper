#!usr/local/env python

import numpy as np
from scipy.integrate import ode


def runga(func, xint, t0, tf, dt):
    """ Performs integration using Runge-Kutta scheme, which is a
        4th-5th order accurate computation.
    """
    # ode: generic class for numeric integrators
    #     r.t: current time
    #     r.y: current variable values; values of state vector
    r = ode(func).set_integrator('dopri5')
    r.set_initial_value(xint, t0)
    time = []
    sol = []
    while r.successful() and r.t <= tf:
        r.integrate(r.t+dt)
        time.append(r.t)
        sol.append(r.y)
    time = np.asarray(time, dtype=float)
    sol = np.asarray(sol, dtype=float)

    # sol is the solution vector, usually in the form of
    #   sol[0] = x, sol[1] = xdot

    return time, sol


def smd(t, x):

    xp = np.zeros((2, 1), dtype=float)
    # print t
    xp[0] = x[1]
    xp[1] = forcing_thomson(t)/m - b/m*x[1] - k/m*x[0]

    return xp


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


if __name__ == "__main__":

    # thomson example 4.8.2 parameters
    k = 100.
    b = 8.0
    m = 2.0

    xinit = np.array(([0., 0.]), dtype=float)
    tzero = 0.
    tfinal = 100.  # [seconds]
    deltat = 0.05
    time, solvec = runga(smd, xinit, tzero, tfinal, deltat)
