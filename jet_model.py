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

#from model import tjet_distr

def generate_jet_distr(Niter, mean_Njet, alpha, tmin):
    Njet_array = stats.poisson.rvs(mean_Njet, size=Niter)
    total_Njet = np.sum(Njet_array)
    cum_Njet1 = np.cumsum(Njet_array)
    cum_Njet2 = np.array(list(cum_Njet1)[1:] + [total_Njet])
    #tjets = tjet_distr(alpha=alpha, tmin=tmin).rvs(size=total_Njet)
    # Wow this is 1000 times faster LOL
    tjets = tmin * (np.random.uniform(size=total_Njet))**(1/(1-alpha))
    total_tjets = [np.sum(tjets[i1:i2]) for i1, i2 in zip(cum_Njet1, cum_Njet2)]
    return np.array(total_tjets)
if __name__=="__main__":
    out = generate_jet_distr(100, 3, 2.2, 1)
    
