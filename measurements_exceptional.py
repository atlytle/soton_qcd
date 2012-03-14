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

def bilinear_LambdaJK(Data):
    pass

# Fourquark Stuff.
def amp_KmK(prop, fourquark):
    iprop = inv(prop)
    ipropT = np.transpose(iprop)
    KmK = fourquark - np.transpose(fourquark, (0, 3, 2, 1))  # i j k l
    tmp1 = tensordot(iprop, KmK, [1,0])    # i' j k l
    tmp2 = tensordot(ipropT, tmp1, [1,1])  # j' i' k l
    tmp3 = tensordot(iprop, tmp2, [1,2])   # k' j' i' l
    tmp4 = tensordot(ipropT, tmp3, [1,3])  # l' k' j' i'
    return np.transpose(tmp4)              # i' j' k' l'

def amputate_fourquark(prop_list, fourquark_array):
    "Amputate fourquark Green's functions."
    N = len(prop_list)
    assert N == len(prop_list) == len(fourquark_array)
    prop = sum(prop_list)/N
    fourquark = sum(fourquark_array)/N
    return [amp_KmK(prop, fourquark[g]) for g in range(16)]

from measurements import G_VVpAA, G_VVmAA, G_SSmPP, G_SSpPP, G_TT
from measurements import proj_g, fourquark_proj_g, F_gg

def fourquark_Zs(Data):
    "Z-factors corresponding to fourquark operators in (X,Y)-schemes."

    amputated = amputate_fourquark(Data.prop_list, Data.fourquark_array)
    norm = (Data.V)**3
    VpA = Data.Lambda_VpA
    Data.fourquark_Lambda = (fourquark_proj_g(amputated).real)*norm
    Data.Zinv = dot(Data.fourquark_Lambda, inv(F_gg))/(VpA)**2

