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
sigma1 = 0.01
sigma2 = 0.02
astar = sigma2**2


def f(x): 
    return X0+bstar+(astar)*x/2 + lam*t/(nu-x) - np.log(alpha)/x

   
X = np.linspace(1./nu,nu-1./nu,1000)
plt.plot(X,f(X))
print("BorneSup VaR = ")
np.min(f(X))
