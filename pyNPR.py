import random
import math
import numpy as np
from numpy import array, trace, dot, tensordot
from numpy.linalg import inv

import domain_wall as dw
from domain_wall import Gc

# Helper Functions
def mmap(x):
    "1 --> '1', -1 --> 'm1'"
    if type(x) is int or type(x) is float:
        if x >= 0:
            return str(x)
        else:
            return 'm' + str(-x)
    else:
        raise TypeError(str(x) + ' must be of type int or float')

def pstring(p):
    "(-1, 0, 1, 0) --> 'm1010'"
    if len(p) == 4:
        temp = map(mmap, p)
        return ''.join(temp)
    else:
        raise TypeError(str(p) + ' must be of length 4.')

def in2out(p):
    '''(-x, 0, x, 0) -> (0, x, x, 0)'''
    if len(p) == 4 and p[0] == -p[2] and p[1] == p[3] == 0:
        x = p[2]
        return (0, x, x, 0)
    else:
        raise TypeError(str(p) + ' must be of form (-x, 0, x, 0).')

def sgn(g):
    if g >= 0:
        return 1
    if g < 0:
        return -1
    else:
        raise TypeError('{0} is non-numeric'.format(g))

def chop(z, tol = .0001):
    re = abs(z.real) > tol and z.real
    im = abs(z.imag) > tol and z.imag
    return complex(re, im)

chopv = np.vectorize(chop)

# Class Definitions.
class Data:
    def __init__(self, m, p, tw, gauge_list):
        # validate data here?
        self.m, self.p, self.tw, self.gauge_list = m, p, tw, gauge_list
        self.p2 = in2out(p) # you should give these better names
        self.loaded = False

        # contingent on subclass, wrap in try?
        self.ap = dw.ap(p, tw, self.L, self.T)
        self.ap2 = dw.ap(self.p2, tw, self.L, self.T)
        self.aq = dw.aq(self.ap, self.ap2)  # ap - ap2
        self.apSq = dw.inner(self.ap)  # (ap)^2
        self.mu = dw.mu(self.ap, self.a)


    def load(self):
        '''Load data structures stored on disk.'''
        # validation? checks?
        # also might you supply more structure, esp in bilinear_array?
        if not self.loaded:
            self.inprop_list = [np.load(self.prop_location(gf, "in"))
                                for gf in self.gauge_list]
            self.outprop_list = [np.load(self.prop_location(gf, "out"))
                                 for gf in self.gauge_list]
            self.bilinear_array = [array([self.bilin_load(gf, gamma) 
                                   for gamma in range(16)])
                                   for gf in self.gauge_list]
            self.fourquark_array = [array([self.fourquark_load(gf, gamma)
                                    for gamma in range(16)])
                                    for gf in self.gauge_list]
            self.loaded = True
        else:
            pass
    
    def prop_location(self, gf, io):
        rest = '.{0}.q{1}.npy'.format(gf, io)
        return self.root + rest

    def bilin_location(self, gf, gamma):
        rest = '.{0}.g{1}.npy'.format(gf, gamma)
        return self.root + rest

    def fourquark_location(self, gf, gamma):
        rest = '.{0}.fq{1}.npy'.format(gf, gamma)
        return self.root + rest

    def bilin_load(self, gf, gamma):
        '''Hack to return the correct bilinear correlation function.'''
        bilin = np.load(self.bilin_location(gf, 15-gamma))
        # should factor this out
        sgn_hack = trace(reduce(dot, 
                         [inv(Gc[gamma]), Gc[15], Gc[15-gamma]]))
        return sgn_hack*dot(Gc[15], bilin)/12

    def fourquark_load(self, gf, gamma):
        return np.load(self.fourquark_location(gf, gamma))

    def clear(self):
        '''Delete references to memory intensive data structures.'''
        del self.inprop_list
        del self.outprop_list
        del self.bilinear_array
        del self.fourquark_array
        self.loaded = False

class DSDR_Data(Data):

    L, T = 32., 64.
    V = (L**3)*T
    a = 1/1.3749  # 1/GeV
    mres = .0018347

    def __init__(self, m, p, tw, gauge_list):

        Data.__init__(self, m, p, tw, gauge_list)

        self.root = '/Users/atlytle/Documents/AuxDetNPR/'\
                    'm{0}/npy/gfmomNE_{1}_{2}_{0}_tw{3}'.format(
                     str(m), pstring(p), pstring(self.p2), mmap(tw))
 
class IWf_Data(Data):
    
    L, T = 32., 64.
    V = (L**3)*T
    a = 1/2.282  # 1/GeV
    mres = .0006664

    def __init__(self, m, p, tw, gauge_list):

        Data.__init__(self, m, p, tw, gauge_list)

        self.root = '/Users/atlytle/Documents/IwasakiNPR/32x64/'\
                    'm{0}/gfmomNE_{1}_{2}_{0}_tw{3}'.format(
                     str(m), pstring(p), pstring(self.p2), mmap(tw))
 
class IWc_Data(Data):

    L, T = 24., 64.
    V = (L**3)*T
    a = 1/1.730  # 1/GeV
    mres = .003152

    def __init__(self, m, p, tw, gauge_list):

        Data.__init__(self, m, p, tw, gauge_list)

        self.root = '/Users/atlytle/Documents/IwasakiNPR/24x64/'\
                    'm{0}/gfmomNE_{1}_{2}_{0}_tw{3}'.format(
                     str(m), pstring(p), pstring(self.p2), mmap(tw))
 
