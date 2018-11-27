#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:16:49 2018

@author: louiseanfray
"""

import numpy as np
import matplotlib.pyplot as plt

t = 2
X0 = 20.
lam = 4
nu = 2
b1 = 4.
b2 = 5.
sigma1 = 0.1
sigma2 = 0.3

NB_SIMULATIONS = 100
NB_POINTS = 1000*t

def echantillonTemps(t): 
    return np.linspace(0,t,NB_POINTS+1)



#Simulation d'un processus de poisson composé

def nombreSauts(lam,t):
    return np.random.poisson(lam*t)

def instantsSauts(n,lam,t):
    return np.sort(t*np.random.rand(n)).tolist()


def listeN(lam,t):
    N = NB_SIMULATIONS*[0]
    for i in range(NB_SIMULATIONS): 
        N[i] = nombreSauts(lam,t)
    return N

def listeT(N,lam,t):
    T = NB_SIMULATIONS*[0]
    for i in range(NB_SIMULATIONS): 
        T[i] = instantsSauts(N[i],lam,t)
    return T
    
simulationsN = listeN(lam,t)
simulationsT = listeT(simulationsN,lam,t)

N = nombreSauts(lam,t)
T = instantsSauts(N,lam,t)
    
    
def actif(t,X0,b1,b2,sigma1 ,sigma2, lam, nu):
    sigma = np.random.uniform(sigma1,sigma2)
    b = np.random.uniform(b1,b2)
    
    N = nombreSauts(lam,t)
    T = instantsSauts(N,lam,t)
                       
    temps = echantillonTemps(t)
    
    #processus de poisson
    L=pertes(T,N)
    
    return [X0+b*temps[i] - L [i] for i in range(NB_POINTS+1)] + sigma*np.cumsum((t/NB_POINTS+1)*np.random.randn(NB_POINTS+1))
           

def pertes(T,N):
    L = [0.]*(NB_POINTS+1)
    T = T + [t]
    taillesSauts =  -1/nu*np.log(np.random.rand(N+1))
    taillesSauts = taillesSauts + [0.]
    taillesSauts = taillesSauts.tolist()
    
    temp = 0.
    for k in range(len(T)-1):
        for i in range(int(T[k]*1000),int(T[k+1]*1000)+1):
            L[i] = temp + taillesSauts[k]
        temp = temp + taillesSauts[k]  
    return L


simulationsX = NB_SIMULATIONS*[0]
for i in range(NB_SIMULATIONS):
    simulationsX[i] = actif(t,X0, b1,b2,sigma1 ,sigma2, lam, nu)
    plt.plot(echantillonTemps(t),simulationsX[i])

def Ruine(simulationsX):
    a=0
    for i in range(NB_SIMULATIONS):
        B = False
        X = simulationsX[i]
        for x in X:
            if x<0:
                B = True
                break
        if B:
            a+=1
    return(a/float(NB_SIMULATIONS))

def sup(X):
    return np.max(X)

#def Var(alpha,X):
    



        
                       
                       



 
