import math
import numpy as np

from math import floor
from itertools import product


# Define gamma matrices.

gamma = [None, None, None, None]

gamma[0] = np.array([[0, 0, 0, 1j],
                     [0, 0, 1j, 0],
                     [0, -1j, 0, 0],
                     [-1j, 0, 0, 0]])

gamma[1] = np.array([[0, 0, 0, -1],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [-1, 0, 0, 0]])

gamma[2] = np.array([[0, 0, 1j, 0],
                     [0, 0, 0, -1j],
                     [-1j, 0, 0, 0],
                     [0, 1j, 0, 0]])

gamma[3] = np.array([[0, 0, 1, 0],
                     [0, 0, 0, 1],
                     [1, 0, 0, 0],
                     [0, 1, 0, 0]])

id4 = np.identity(4, dtype=complex)


# QDP enumeration.

def hc(M):
    "Hermitian conjugate."
    return np.conjugate(np.transpose(M))

def dtob4(d):
    "Convert a decimal integer 0..15 to binary."
    result = [None, None, None, None]
    if d in range(16):
        for i in range(4):
            r = d % 2
            result[i] = r
            d = (d-r)/2
        return result
    else:
        raise ValueError(str(d) + ' not in 0..15')

def one_or_M(M, num):
    if M.shape == (4, 4): 
        if num is 0:
            return id4
        if num is 1:
            return M
        else:
            raise ValueError("num must be 0 or 1")
    else:
        raise TypeError("M must be a 4x4 array")

def Gamma(n):
    binary = dtob4(n)
    gammas = [gamma[0], gamma[1], gamma[2], gamma[3]]
    gammas = map(one_or_M, gammas, binary)  # gamma[0]^n[0]..gamma[3]^n[3]
    return reduce(np.dot, gammas)

G = [Gamma(n) for n in range(16)]

# Color un-mixed projectors.
Gc = [np.kron(np.identity(3), G[n]) for n in range(16)]

# Color mixed projectors.
def switch(i):
    "Switch between spin-color basis and tensor basis."
    return (i%4, i/4)

#can mixArray and qqMixArray be integrated?
def mixArray(g):
    "Color-mixed gamma projectors."

    GG = G[g]
    kd = np.identity(3)

    def newMixProj(ii, jj, kk, ll):
        "Spin-color basis color-mixed projectors."
        i, a = switch(ii)
        j, b = switch(jj)
        k, c = switch(kk)
        l, d = switch(ll)
        return GG[i][j]*GG[k][l]*kd[a][d]*kd[b][c] #ij -> ji?

    r = np.zeros((12, 12, 12, 12), complex) # !
    for i, j, k, l in product(range(12), repeat=4):
        r[i][j][k][l] = newMixProj(i, j, k, l)
    return r

def qqMixArray(aq, pseudo=False):
    "Color-mixed qslash x qslash projectors."
    # optimize!
    q = slash_nc(aq)
    if pseudo:
        q = np.dot(q, G[15])
    kd = np.identity(3)

    def qqMixProj(ii, jj, kk, ll):
        #global q, kd
        i, a = switch(ii)
        j, b = switch(jj)
        k, c = switch(kk)
        l, d = switch(ll)
        return q[j][i]*q[l][k]*kd[d][a]*kd[b][c]

    r = np.zeros((12, 12, 12, 12), complex) # !
    for i, j, k, l in product(range(12), repeat=4):
        r[i][j][k][l] = qqMixProj(i, j, k, l)
    return r

def sigmaMixArray(aq):
    "Color-mixed sigma.q x sigma.q projectors."
    sdq = sigma_dot_q(aq, color=False)
    kd = np.identity(3)

    def sigmaMixProj(ii, jj, kk, ll):
        #global q, kd
        i, a = switch(ii)
        j, b = switch(jj)
        k, c = switch(kk)
        l, d = switch(ll)
        sdq_dot_sdq=sum([sdq[mu][j][i]*sdq[mu][l][k] for mu in range(4)])
        return (sdq_dot_sdq)*kd[d][a]*kd[b][c]

    r = np.zeros((12, 12, 12, 12), complex) # !
    for i, j, k, l in product(range(12), repeat=4):
        r[i][j][k][l] = sigmaMixProj(i, j, k, l)
    return r


# Momentum definitions.
def sgn0(x):
    "Sign function w/ 0 -> 0."
    if x > 0:
        return 1
    if x < 0:
        return -1
    if x == 0:
        return 0
    else:
        raise TypeError('{0} is not comparable to 0'.format(x))

def ap(p, tw, L=32, T=64):
    "Momentum vector in lattice units."
    x = math.pi/L
    y = math.pi/T
    sp = map(sgn0, p)
    return (2*x*p[0] + x*tw*sp[0], 2*x*p[1] + x*tw*sp[1], 
            2*x*p[2] + x*tw*sp[2], 2*y*p[3] + y*tw*sp[3])

def slash(p):
    "Gamma_mu ap^mu in spin x color basis."
    g = (Gc[1], Gc[2], Gc[4], Gc[8])
    return sum([g[mu]*p[mu] for mu in range(4)])

def slash_nc(p):
    "Gamma_mu ap^mu in spin basis."
    g = (G[1], G[2], G[4], G[8])
    return sum(g[mu]*p[mu] for mu in range(4))

def sigma(mu, nu, color=False):
    "sigma^{mu nu} in spin(-color) basis."
    g = (G[1], G[2], G[4], G[8])
    if color:
        g = (Gc[1], Gc[2], Gc[4], Gc[8])

    return (1/2.)*(np.dot(g[mu], g[nu]) - np.dot(g[nu], g[mu]))

def sigma5(mu, nu, color=False):
    "sigma^{mu nu} gamma_5 in spin(-color) basis."
    g5 = G[15]
    if color:
        g5 = Gc[15]
    return np.dot(sigma(mu,nu,color), g5)
    
def sigma_dot_q(aq, color=False):
    "Four-list sigma^{mu nu} q_{nu} in spin(-color) basis."
    sdq = [0, 0, 0, 0]
    for mu in range(4):
        sdq[mu] = sum([sigma(mu, nu, color)*aq[nu] for nu in range(4)])
    return sdq

def sigma_dot_q5(aq, color=False):
    "Four-list q_nu sigma^{mu nu} gamma_5."
    sdq5 = [0, 0, 0, 0]
    for mu in range(4):
        sdq5[mu] = sum([sigma5(mu, nu, color)*aq[nu] for nu in range(4)])
    return sdq5

def aq(ap1, ap2):
    "Return tuple ap1 - ap2."
    return tuple([ap1[i] - ap2[i] for i in range(4)])

def inner(vec):
    "Inner product of vector with itself."
    return sum([x*x for x in vec])

def mu(ap, a):
    "Magnitude of vector ap in physical units."
    psq = inner(ap)/(a*a)
    return math.sqrt(psq)
