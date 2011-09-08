import math, sys
import numpy as np
from scipy.integrate import odeint

def Beta(a, mu2, Nf, Nc=3):
    '''Return da/dln(mu^2) = Beta(a).'''
    B0 = (11./3)*Nc - (2./3)*Nf
    CF = (Nc*Nc-1)/(2.*Nc)  # Casimir factor.
    B1 = (34./3)*Nc*Nc - (10./3)*Nc*Nf - 2*CF*Nf
    return -(B0*a**2 + B1*a**3)

# Conversion functions.
a_mu2 = lambda alpha, mu: (alpha/(4*math.pi), mu*mu)
alpha_mu = lambda a, mu2: (4*math.pi*a, math.sqrt(mu2))

def alpha_s(mu, mu0, alpha0, Nf, Nc=3):
    a0, mu2 = a_mu2(alpha0, np.linspace(mu0, mu, 1000))
    B = lambda a, mu2: Beta(a, mu2, Nf, Nc)
    a = odeint(B, a0, np.log(mu2))
    return 4*math.pi*a
    
MZ = 91.1876  # GeV
Mb = 4.19  # GeV
Mc = 1.27  # GeV
alpha_MZ = .1184  # Initial condition.

def main():
    print "MZ = " + str(MZ)
    print "Mb = " + str(Mb)
    print "alpha_s(MZ) (Nf=5) = " + str(alpha_MZ)
    alpha_Mb = alpha_s(Mb, MZ, alpha_MZ, Nf=5)[-1,0]
    print "alpha_s(Mb) (Nf=5) = " + str(alpha_Mb)
    alpha_Mc = alpha_s(Mc, Mb, alpha_Mb, Nf=4)[-1,0]
    print "alpha_s(Mc) (Nf=4) = " + str(alpha_Mc)
    alpha_2GeV = alpha_s(2., Mc, alpha_Mc, Nf=3)[-1,0]
    print "alpha_s(2 GeV) (Nf=3) = " + str(alpha_2GeV)
    alpha_3GeV = alpha_s(3., Mc, alpha_Mc, Nf=3)[-1,0]
    print "alpha_s(3 GeV) (Nf=3) = " + str(alpha_3GeV)
    return 0

if __name__ == "__main__":
    sys.exit(main())
