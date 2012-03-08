import numpy as np
from numpy import array, trace, dot, tensordot
from numpy.linalg import inv

import domain_wall as dw
from domain_wall import Gc
from pyNPR import sgn
from measurements import JKsample, JKsigma, bootstrap_sample

# Measurements

def amputate_bilinears_e(prop_list, bilinear_array):
    N = len(prop_list)
    assert N == len(prop_list) == len(bilinear_array)
    prop_inv = inv(sum(prop_list)/N)
    bilinears = sum(bilinear_array)/N
    return [reduce(dot, [prop_inv, bilinears[g], prop_inv])
            for g in range(16)]

def bilinear_Lambdas(Data):
    amputated = amputate_bilinears_e(Data.prop_list, Data.bilinear_array)
    norm = Data.V/12
    Lambda =lambda x, y: (trace(dot(dw.hc(x), y)).real)*norm
    Data.Lambda = map(Lambda, Gc, amputated)

    # For the (X, g) schemes.
    Data.Lambda_VpA = (Data.Lambda[1] + Data.Lambda[2] + Data.Lambda[4] +
                       Data.Lambda[8] + Data.Lambda[14] + Data.Lambda[13] +
                       Data.Lambda[11] + Data.Lambda[7])*(1./8)   
    Data.Lambda_V = (Data.Lambda[1] + Data.Lambda[2] + Data.Lambda[4] +
                       Data.Lambda[8])*(1./4)   
    Data.Lambda_A = (Data.Lambda[14] + Data.Lambda[13] + Data.Lambda[11] +
                       Data.Lambda[7])*(1./4)
