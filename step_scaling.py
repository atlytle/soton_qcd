from numpy import dot, sqrt
from numpy.linalg import inv
import numpy as np

def sigma(Zf, Zi):
    "Z_f/Z_i"
    if Zi.shape == Zf.shape == (1,):
        return Zf/Zi
    else:
        return np.dot(Zf, np.linalg.inv(Zi))
'''
def step_scale(Data_f, Data_i):
    "Sigma in all four schemes."
    Data_f.step_scale = {}
    for scheme in 'gg', 'gq', 'qg', 'qq':
        Data_f.step_scale[scheme] = sigma(Data_f.fourquark_Zs[scheme],
                                          Data_i.fourquark_Zs[scheme])
'''

def step_scale(Data_f, *Dat_i):
    '''Sigma for multiple matching points mu0.'''
    Data_f.step_scale = {}
    for scheme in 'gg', 'gq', 'qg', 'qq':
        Data_f.step_scale[scheme] = [sigma(Data_f.fourquark_Zs[scheme],
                                           d.fourquark_Zs[scheme])
                                     for d in Dat_i]
        if len(Dat_i) == 1:
            Data_f.step_scale[scheme] = Data_f.step_scale[scheme][0]
        
#  The formula is off, see soton.tex
def step_scale_sigma(Data_f, Data_i):
    '''Naive error propagation for step-scaling functions.'''
    Data_f.step_scale_sigma = {}
    for scheme in 'gg', 'gq', 'qg', 'qq':    
        Zf = Data_f.fourquark_Zs[scheme]
        Zi = Data_i.fourquark_Zs[scheme]
        dZf = (Data_f.fourquark_sigmaJK[scheme])**2
        tmp = [inv(Zi), Data_i.fourquark_sigmaJK[scheme], inv(Zi)]
        dZi_inv = reduce(dot, tmp)**2
        dSS = sqrt(dot(dZf, inv(Zi)) + dot(Zf, dZi_inv))
        Data_f.step_scale_sigma[scheme] = dSS

def step_scale_JK(Data_f, Data_i):
    pass

def do_ss(denominator):
    "Return a function that calculates Z/denominator."
    return lambda d: step_scale(d, denominator)

def do_ssJK(denominator):
    "Return a function that calculates sigma_{Z/denominator}."
    return lambda d: step_scale_sigma(d, denominator)
