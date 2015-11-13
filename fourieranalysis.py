#!usr/local/env python
# use '#!usr/local/bin/anaconda/bin/python' for specific path execution
# try '#!usr/local/bin python' also
#
# Computes the Fourier transform of a signal and filters it.
#
# last update: jbp3, 20151029

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as scisig
from numpy.fft import fft, fftshift, ifft

sr = 400.0  # Hz, sampling rate
dt = 1./sr  # timestep size


def fourieranalysis(Fdata):
    """ Computes the Fourier transform of input signal Fdata.
        Returns the frequency spectrum coordinates in 's' and
        the spectra's real and imaginary parts, centered around
        s = 0 Hz, in 'dftFshift'
    """
    N = len(Fdata)  # number of samples

    ds = 1./(N*dt)  # frequency step, Hz
    if ((N % 2) == 0):
        s = np.arange(N)*ds-N*ds/2.
    else:
        s = np.arange(N)*ds-(N-1)*ds/2.

    dftFshift = fftshift(fft(Fdata)/N)

    return s, dftFshift


def rectfilterdft(Fdata):

    N = len(Fdata)
    ds = 1./(N*dt)
    
    fthresh = 1.  # Hz, freq threshold for low pass filter
    fn = int(np.floor(fthresh/ds))
    filt = np.zeros((N), dtype=float)
    filt[0:fn] = 1.
    filt[N-fn:N] = 1.
    # take the FT of the data
    dft = fft(Fdata)/N
    # filter data with low-pass filter
    dftfilt = filt*dft
    # inverse fft to get data in time-domain
    filtdata = ifft(dftfilt*N)

    return filtdata


def plotdft(Fdata):

    goldratio = 1.61803398875  # golden ratio
    plotwidth = 10.  # inches
    plt.figure(dpi=75, facecolor='0.8',
               figsize=(plotwidth, plotwidth/goldratio))
    ax = plt.subplot(211)
    s, Fs_dftshift = fourieranalysis(Fdata)
    plt.plot(s, Fs_dftshift.real, label=r'$\mathcal{Re}(F(s))$')
    plt.plot(s, Fs_dftshift.imag, label=r'$\mathcal{Im}(F(s))$')
    
    #  plt.xlabel(r'frequency, [Hz]')
    plt.ylabel(r'$\mathcal{F}(f(t))$')
    plt.suptitle('DFT', fontsize='medium')
    plt.xlim(-np.max(s), np.max(s))
    # plt.xlim(0., 10.)
    plt.grid()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    plt.legend(loc='center left',
               bbox_to_anchor=(1.01, 0.50),
               borderaxespad=0., labelspacing=0.1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')

    ax = plt.subplot(212)
    Fs_dftphase = np.arctan(np.imag(Fs_dftshift)/np.real(Fs_dftshift))
    plt.plot(s, Fs_dftphase, label=r'$\phi=\frac{\mathcal{Im}}{\mathcal{Re}}$')
    plt.xlabel(r'frequency, [Hz]')
    plt.ylabel(r'phase, $\phi$')
    plt.xlim(-np.max(s), np.max(s))
    plt.grid()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
    plt.legend(loc='center left', \
               bbox_to_anchor=(1.05,0.50), \
               borderaxespad=0., labelspacing=0.1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize='small')
    plt.savefig('dft.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == "__main__":

    freq = 2.  # Hz
    tf = 10.
    t = np.linspace(-tf/2., tf/2., num=tf*sr)
    y = np.cos(2.*np.pi*freq*t)

    plotdft(y)
