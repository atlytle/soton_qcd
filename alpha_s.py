import math
import numpy as np
from scipy.integrate import odeint

def Beta(a, mu2, Nf=3, Nc=3):
    '''Return da/dln(mu^2) = Beta(a).'''
    B0 = (11./3)*Nc - (2./3)*Nf
    CF = (Nc*Nc-1)/(2.*Nc)  # Casimir factor.
    B1 = (34./3)*Nc*Nc - (10./3)*Nc*Nf - 2*CF*Nf
    #print B0, CF, B1
    return -(B0*a**2 + B1*a**3)

# Conversion functions.
a_mu2 = lambda alpha, mu: (alpha/(4*math.pi), mu*mu)
alpha_mu = lambda a, mu2: (4*math.pi*a, math.sqrt(mu2))

def alpha_s(mu, mu0, alpha0):
    a0, mu2 = a_mu2(alpha0, np.linspace(mu0, mu, 100))
    res = odeint(Beta, a0, mu2)
    return 4*math.pi*res
    

MZ = 74  # GeV
Mb = 5
Mc = 1
#mu2 = linspace

def main():
    pass
