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
    fjet = 0.1
    Mgas = 10**7 # Msun
    SFE = 0.002 # Ratio of Mstar/Mgas
    yFe = 0.1 # Msun
    yEu_rate = 1/1000 * .006 #10**-5 * 0.002 # Msun
    
    print("fjet = {}, Mgas = {:.1e}, SFE = {}".format(fjet,Mgas,SFE))
    print("yFe = {}, yEu at 300s = {:.2e}".format(yFe, yEu_rate*300))
    Mstar = Mgas * SFE
    SN_to_Mass = 1/0.009 # Salpeter IMF
    Mass_to_SN = 0.009 # Salpeter IMF
    
    #N_SN = 100
    #N_jet = int(N_SN * fjet)
    
    
    start = time.time()
    np.random.seed(2304897)
    Niter = 1000
    default_tjet = tjet_distr(tmin=1.)
    
    #MFe = N_SN*yFe
    #print("[Fe/H] = {:.2f}".format(MFeMH_to_FeH(MFe,Mgas)))
    #print("Stellar Mass = {:.2e}".format(SN_to_Mass*N_SN))
    #print("Stellar Mass/Mgas = {:.3f}".format(SN_to_Mass*N_SN/Mgas))
    NSN_array = np.zeros(Niter)
    Njet_array = np.zeros(Niter)
    MFe_array = np.zeros(Niter)
    MEu_array = np.zeros(Niter)
    for i in range(Niter):
        N_SN = stats.poisson.rvs(Mgas*SFE*Mass_to_SN)
        MFe_array[i] = N_SN*yFe
        MEu = 0.0
        N_jet = stats.poisson.rvs(N_SN*fjet)
        for j in range(N_jet):
            MEu += yEu_rate * default_tjet.rvs()
        MEu_array[i] = MEu
        NSN_array[i] = N_SN
        Njet_array[i] = N_jet
    EuFe = MEuMFe_to_EuFe(MEu_array,MFe_array)
    EuFe_min = -2
    EuFe[np.isinf(EuFe)] = EuFe_min
    FeH = MFeMH_to_FeH(MFe_array,Mgas*0.75)
    print("took {:.1f}s".format(time.time()-start))
    
    fig, ax = plt.subplots()
    ax.hist(EuFe,bins="auto")
    print("[Fe/H] = {:.2f} +/- {:.2f}".format(np.median(FeH), np.diff(np.percentile(FeH,[50,84]))[0]))
    print("[Eu/Fe] = {:.2f} +/- {:.2f}".format(np.median(EuFe), np.diff(np.percentile(EuFe,[50,84]))[0]))
    print("N_SN = {:.2f} +/- {:.2f}".format(np.median(NSN_array), np.diff(np.percentile(NSN_array,[50,84]))[0]))
    print("N_jet = {:.2f} +/- {:.2f}".format(np.median(Njet_array), np.diff(np.percentile(Njet_array,[50,84]))[0]))
    
    plt.show()
