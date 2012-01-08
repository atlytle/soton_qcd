import math, sys, itertools
import numpy as np
from numpy import array, dot

# dwf-function-defs.nb
def phat(p):
    hat = lambda p: 2*math.sin(p/2.)
    return map(hat, p)

def pbar(p):
    bar = lambda p: math.sin(p)
    return map(bar, p)

def W(p, M):
    ph = phat(p)
    return 1 - M + dot(ph, ph)/2.

def ea(p, M):
    w = W(p, M)
    pb = pbar(p)
    pb_sq = dot(pb, pb)
    ch = (1 + w*w + pb_sq)/(2.*abs(w))
    return ch + math.sqrt(ch*ch-1)

def xx(p, M):
    w = W(p, M)
    pb = pbar(p)
    pb_sq = dot(pb, pb)
    return (1 + w*w + pb_sq)/(2.*abs(w))

def yy(p, M):
    w = W(p, M)
    pb = pbar(p)
    pb_sq = dot(pb, pb)
    return 1 + w*w + pb_sq

def zz(p, M):
    w = W(p, M)
    y = yy(p, M)
    return y/2. + math.sqrt(y*y/4. - w*w)

# free-dwf-propagator.nb
g = [None for x in range(6)]
g[0] = array([[0, 0, 1, 0],
              [0, 0, 0, 1],
              [1, 0, 0, 0],
              [0, 1, 0, 0]])
g[1] = array([[0, 0, 0, -1j],
              [0, 0, -1j, 0],
              [0, 1j, 0, 0],
              [1j, 0, 0, 0]])
g[2] = array([[0, 0, 0, -1],
              [0, 0, 1, 0],
              [0, 1, 0, 0],
              [-1, 0, 0, 0]])
g[3] = array([[0, 0, -1j, 0],
              [0, 0, 0, 1j],
              [1j, 0, 0, 0],
              [0, -1j, 0, 0]])
g[4] = g[0]
g[5] = reduce(dot, [g[1], g[2], g[3], g[4]])
id4 = np.identity(4)

Pp = (id4 + g[5])/2.
Pm = (id4 - g[5])/2.
Pstat = (id4 + g[0])/2.

def ap(n_tuple, L, T):
    n = n_tuple
    pi = math.pi
    return (2*pi*n[0]/T, 2*pi*n[1]/L, 2*pi*n[2]/L, 2*pi*n[3]/L)

def four_dot(a, b):
    '''Dot product of four-vectors.'''
    return sum([a[i]*b[i] for i in range(4)])

def pslash(p):  # Note different index convention!
    return sum([p[i]*g[i] for i in range(4)])

def mompropLs(p, m, M, Ls):
    '''Free momentum space propagator, S(p).'''
    pbar_sl = pslash(pbar(p))
    w = W(p, M)
    z = zz(p, M)
    owe = 1 - w*w/z
    ea = z/w #dif from ea(...)?
    fac = (z-w*w/z)/(ea**(2*Ls)-1) #N=Ls?
    f = 1 - z - m*m*owe + fac*(2*m*ea**Ls - 1 - m*m)
    return (1j*pbar_sl + (-m*owe + fac*(ea**Ls - m))*id4)/f 

def propagatorLs(x, m, M, L, T, Ls):
    "S(x, 0)"
    result = np.zeros((4,4), 'complex')
    for n in itertools.product(range(0,T), range(0,L), range(0,L), range(0,L)):
      #print ap(n, L, T)
      p = ap(n, L, T)
      phase = four_dot(p, x) #just use dot numpy arrays?
      eipx = complex(math.cos(phase), math.sin(phase))
      result += mompropLs(p, m, M, Ls)*eipx
    return result/(L*L*L*T)

def main():
    L, T, M, m, Ls = 4, 8, 1.8, 1.0, 16
    return 0

if __name__ == "__main__":
    sys.exit(main())
