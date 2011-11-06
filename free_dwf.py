import math
from numpy import dot

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
    ch = 1 + w*w + pb_sq/(2.*abs(w))
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
def pslash(p):
    pass

def prop(p, m, M, Ls):
    '''Free momentum space propagator, S(p).'''
    pass

def prop_x(x, p, m, M, Ls):
    "S(x, p)"
    pass
        

    
