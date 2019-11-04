#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import glob, os, sys, time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii
import seaborn as sns
from scipy import stats

from alexmods import read_data as rd


def MEuMFe_to_EuFe(MEu, MFe):
    muEu = 152.
    muFe = 56.0
    logNEu_NFe = np.log10(MEu*muFe/(muEu*MFe))
    return logNEu_NFe - (0.52-7.5)
def MFeMH_to_FeH(MFe, MH):
    muFe = 56.0
    return np.log10(MFe/muFe/MH) - 7.50 + 12

class tjet_distr(stats.rv_continuous):
    """ This is really slow """
    def __init__(self, alpha=4.2, tmin=13.0):
        super().__init__()
        self.alpha = alpha
        self.tmin = tmin
    def _cdf(self, t):
        t = np.ravel(t)
        out = np.zeros_like(t)
        ii = t > self.tmin
        out[ii] = 1.0 - (t[ii]/self.tmin)**(1-self.alpha)
        return out
    # alpha=4.2,tmin=13.0

if __name__ == "__main__":
    fjet = 0.03
    N_SN = 100
    
    Mgas = 10**7 # Msun
    yFe = 0.25 # Msun
    yEu_rate = 1/300 * .006 #10**-5 * 0.002 # Msun
    
    SN_to_Mass = 1/0.009 # Salpeter IMF
    Mass_to_SN = 0.009 # Salpeter IMF
    
    #default_tjet = stats.powerlaw(loc=,scale=)
    #fig, ax = plt.subplots()
    #tplot = np.logspace(-1,3)
    #ax.plot(tplot, default_tjet.pdf(tplot))
    #ax.set_xscale('log'); ax.set_yscale('log')
    #plt.show()
    
    N_jet = int(N_SN * fjet)
    
    start = time.time()
    np.random.seed(2304897)
    Niter = 1000
    default_tjet = tjet_distr()
    MFe = N_SN*yFe
    
    print("[Fe/H] = {:.2f}".format(MFeMH_to_FeH(MFe,Mgas)))
    print("Stellar Mass = {:.2e}".format(SN_to_Mass*N_SN))
    print("Stellar Mass/Mgas = {:.3f}".format(SN_to_Mass*N_SN/Mgas))
    MEu_array = np.zeros(Niter)
    for i in range(Niter):
        MEu = 0.0
        this_N_jet = stats.poisson.rvs(N_jet)
        for j in range(this_N_jet):
            MEu += yEu_rate * default_tjet.rvs()
        MEu_array[i] = MEu
    EuFe = MEuMFe_to_EuFe(MEu_array,MFe)
    EuFe_min = -2
    EuFe[np.isinf(EuFe)] = EuFe_min
    print("took {:.1f}s".format(time.time()-start))
    fig, ax = plt.subplots()
    ax.hist(EuFe,bins="auto")
    print("[Eu/Fe] = {:.2f} +/- {:.2f}".format(np.median(EuFe), np.diff(np.percentile(EuFe,[50,84]))[0]))
    plt.show()
