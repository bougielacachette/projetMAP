#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 11:23:42 2018

@author: louiseanfray
"""

import numpy as np
import matplotlib.pyplot as plt

"""
Paramètres du problème
"""
#Seuil de confiance 1-alpha pour la VaR
alpha=0.05

#Horizon temporel
t = 2
NB_POINTS = 1000*t

#Paramètres de l'actif
X0 = 0.
lam = 1
nu = 5
b1 = 0.5
b2 = 1.
bstar= max(0,b1,b2)
sigma1 = 0.15
sigma2 = 0.3
astar = sigma2**2

NB_SIMULATIONS = 1000


def echantillonTemps(t): 
    return np.linspace(0,t,NB_POINTS+1)


"""
Simulation d'un processus de Poisson composé
"""
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

def pertes(T,N):
    L = [0.]*(NB_POINTS+1)
    T = T + [t]
    taillesSauts = np.random.exponential(1./nu,N+1)
    #taillesSauts =  -1/nu*np.log(np.random.rand(N+1))
    taillesSauts = taillesSauts + [0.]
    taillesSauts = taillesSauts.tolist()
    #print(taillesSauts)
    temp = 0.
    for k in range(len(T)-1):
        for i in range(int(T[k]*1000),int(T[k+1]*1000)+1):
            L[i] = temp + taillesSauts[k]
        temp = temp + taillesSauts[k]  
    return L

"""
Prix de l'actif
"""
def actif(t,X0,b1,b2,sigma1 ,sigma2, lam, nu):
    sigma = np.random.uniform(sigma1,sigma2)
    b = np.random.uniform(b1,b2)
    
    N = nombreSauts(lam,t)
    T = instantsSauts(N,lam,t)
                       
    temps = echantillonTemps(t)
    
    #processus de poisson
    L=pertes(T,N)
    
    return [X0+b*temps[i]+L[i] for i in range(NB_POINTS+1)] + sigma*np.cumsum((np.sqrt(1./(NB_POINTS+1))*np.random.randn(NB_POINTS+1)))

simulationsX = NB_SIMULATIONS*[0]
for i in range(NB_SIMULATIONS):
    simulationsX[i] = actif(t,X0, b1,b2,sigma1 ,sigma2, lam, nu)
    plt.plot(echantillonTemps(t),simulationsX[i])
    
"""
Probabilité de ruine
"""
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

"""
VaR expérimentale
"""
def VaR(alpha,simulationsX):
    #sup = [max(0.,- np.min(X)) for X in simulationsX]
    sup = [np.max(X) for X in simulationsX]
    sup = np.sort(sup)[::-1].tolist()
    return(sup[int(alpha*NB_SIMULATIONS)-1])

"""
Test
"""
print("VaR expérimentale = ")
print(VaR(alpha,simulationsX))




        
                       
                       



 
