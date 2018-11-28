#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 23:46:54 2018

@author: bougon
"""

import numpy as np
import matplotlib.pyplot as plt


alpha=0.005

def f(x) :
    return(30+0.0225*x+8/(2-x)-np.log(alpha)/x)
    
    
X = np.linspace(0.2,1.8,10000)

plt.plot(X,f(X))

np.min(f(X)) 