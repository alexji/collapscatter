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

def print_stuff(FeH, EuFe, NSN_array, Njet_array):
    print("[Fe/H] = {:.2f} +/- {:.2f}".format(np.median(FeH), np.diff(np.percentile(FeH,[50,84]))[0]))
    print("[Eu/Fe] = {:.2f} +/- {:.2f}".format(np.median(EuFe), np.diff(np.percentile(EuFe,[50,84]))[0]))
    print("N_SN = {:.2f} +/- {:.2f}".format(np.median(NSN_array), np.diff(np.percentile(NSN_array,[50,84]))[0]))
    print("N_jet = {:.2f} +/- {:.2f}".format(np.median(Njet_array), np.diff(np.percentile(Njet_array,[50,84]))[0]))

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

def run_model(Niter, fjet, Mgas, SFE, yFe, yEu_rate, tmin, alpha):
    print("Niter={}, fjet = {}, Mgas = {:.1e}, SFE = {}".format(Niter, fjet,Mgas,SFE))
    print("yFe = {}, yEu at 300s = {:.2e}".format(yFe, yEu_rate*300))
    Mstar = Mgas * SFE
    SN_to_Mass = 1/0.009 # Salpeter IMF
    Mass_to_SN = 0.009 # Salpeter IMF
    
    start = time.time()
    default_tjet = tjet_distr(tmin=tmin,alpha=alpha)
    
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

    return EuFe, FeH, NSN_array, Njet_array

#if __name__ == "__main__":
def plot_first_model():
    Niter = 1000
    fjet0 = 0.01
    Mgas0 = 10**7 # Msun
    SFE0 = 0.002 # Ratio of Mstar/Mgas
    yFe0 = 0.2 # Msun
    yEu_rate0 = 1/1000 * .006 #10**-5 * 0.002 # Msun
    tmin0 = 1.0
    alpha0 = 2.2
    #EuFe, FeH, NSN_array, Njet_array = run_model(100, fjet, Mgas, SFE, yFe, yEu_rate, tmin, alpha)
    
    np.random.seed(23459870)
    
    halo = rd.load_halo()
    halo["[Ba/Eu]"] = halo["[Ba/H]"] - halo["[Eu/H]"]
    ii = (pd.notnull(halo["[Eu/Fe]"]) & (~halo["uleu"]) & (halo["[Fe/H]"] < -2.5) & (halo["[Ba/Eu]"] < -0.4) & (~halo["ulba"]))
    print("Num stars = {}".format(ii.sum()))
    star_eufe = np.sort(halo["[Eu/Fe]"][ii])
    
    fig, axes = plt.subplots(2,2,figsize=(8,8))
    fig.suptitle("fjet={} logMgas={} SFE={} yFe={}\nyEu/t={:.1e} tmin={} alpha={}".format(
        fjet0, np.log10(Mgas0), SFE0, yFe0, yEu_rate0, tmin0, alpha0))
    ax = axes[0,0]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for tmin in [0.1, 1, 10]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet0, Mgas0, SFE0, yFe0, yEu_rate0, tmin, alpha0)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="tmin "+str(tmin))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    ax = axes[0,1]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for fjet in [0.003, 0.01, 0.03]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet, Mgas0, SFE0, yFe0, yEu_rate0, tmin0, alpha0)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="fjet "+str(fjet))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    ax = axes[1,0]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for alpha in [1.7, 2.2, 3.2]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet0, Mgas0, SFE0, yFe0, yEu_rate0, tmin0, alpha)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="alpha "+str(alpha))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    ax = axes[1,1]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for Mgas in [10**6, 10**7, 10**8]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet0, Mgas, SFE0, yFe0, yEu_rate0, tmin0, alpha0)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="logMgas "+str(np.log10(Mgas)))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    fig.savefig("first_model.png", bbox_inches="tight")
    plt.show()
    
if __name__ == "__main__":
    Niter = 100
    fjet0 = 1.0
    Mgas0 = 10**6 # Msun
    SFE0 = 0.001 # Ratio of Mstar/Mgas
    yFe0 = 0.2 # Msun
    yEu_rate0 = 1/1000 * .006 #10**-5 * 0.002 # Msun
    tmin0 = 0.01
    alpha0 = 2.0
    #EuFe, FeH, NSN_array, Njet_array = run_model(100, fjet, Mgas, SFE, yFe, yEu_rate, tmin, alpha)
    
    np.random.seed(23459870)
    
    halo = rd.load_halo()
    halo["[Ba/Eu]"] = halo["[Ba/H]"] - halo["[Eu/H]"]
    ii = (pd.notnull(halo["[Eu/Fe]"]) & (~halo["uleu"]) & (halo["[Fe/H]"] < -2.5) & (halo["[Ba/Eu]"] < -0.4) & (~halo["ulba"]))
    print("Num stars = {}".format(ii.sum()))
    star_eufe = np.sort(halo["[Eu/Fe]"][ii])
    
    fig, ax = plt.subplots()
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet0, Mgas0, SFE0, yFe0, yEu_rate0, tmin0, alpha0)
    ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size)
    print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    plt.show()
    
def tmp():
    fig, axes = plt.subplots(2,2,figsize=(8,8))
    fig.suptitle("fjet={} logMgas={} SFE={} yFe={}\nyEu/t={:.1e} tmin={} alpha={}".format(
        fjet0, np.log10(Mgas0), SFE0, yFe0, yEu_rate0, tmin0, alpha0))
    ax = axes[0,0]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for tmin in [0.1, 1, 10]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet0, Mgas0, SFE0, yFe0, yEu_rate0, tmin, alpha0)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="tmin "+str(tmin))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    ax = axes[0,1]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for fjet in [0.003, 0.01, 0.03]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet, Mgas0, SFE0, yFe0, yEu_rate0, tmin0, alpha0)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="fjet "+str(fjet))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    ax = axes[1,0]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for alpha in [1.7, 2.2, 3.2]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet0, Mgas0, SFE0, yFe0, yEu_rate0, tmin0, alpha)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="alpha "+str(alpha))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    ax = axes[1,1]
    ax.plot(star_eufe, (np.arange(star_eufe.size)+1)/star_eufe.size, 'k-', label="Stars")
    for Mgas in [10**6, 10**7, 10**8]:
        EuFe, FeH, NSN_array, Njet_array = run_model(Niter, fjet0, Mgas, SFE0, yFe0, yEu_rate0, tmin0, alpha0)
        ax.plot(np.sort(EuFe), (np.arange(EuFe.size)+1)/EuFe.size, label="logMgas "+str(np.log10(Mgas)))
        print_stuff(FeH, EuFe, NSN_array, Njet_array)
    ax.legend(); ax.set_xlim(-2,3); ax.set_ylim(0,1)
    
    fig.tight_layout()
    fig.subplots_adjust(top=.9)
    plt.show()
    
