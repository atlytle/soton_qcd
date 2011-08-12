from numpy import dot, sqrt
from numpy.linalg import inv
import numpy as np

def sigma(Zf, Zi):
    "Z_f/Z_i"
    if Zi.shape == Zf.shape == (1,):
        return Zf/Zi
    else:
        return np.dot(Zf, np.linalg.inv(Zi))

def step_scale(Data_f, Data_i):
    "Sigma in all four schemes."
    Data_f.step_scale = sigma(Data_f.fourquark_Zs, Data_i.fourquark_Zs)
    Data_f.step_scale_q = sigma(Data_f.fourquark_Zs_q, Data_i.fourquark_Zs_q)
    Data_f.step_scale_qg = sigma(Data_f.fourquark_Zs_qg, Data_i.fourquark_Zs_qg)
    Data_f.step_scale_qq = sigma(Data_f.fourquark_Zs_qq, Data_i.fourquark_Zs_qq)


def step_scale_sigma(Data_f, Data_i):
    '''Naive error propagation for step-scaling functions.'''
    # (g, g)
    Zf = Data_f.fourquark_Zs
    Zi = Data_i.fourquark_Zs
    dZf = (Data_f.fourquark_sigmaJK)*(Data_f.fourquark_sigmaJK)
    dZi_inv = reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK, inv(Zi)])*\
              reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK, inv(Zi)])
    dSS = sqrt(dot(dZf, inv(Zi)) + dot(Zf, dZi_inv))
    Data_f.step_scale_sigma = dSS

    # (g, q)
    Zf = Data_f.fourquark_Zs_q
    Zi = Data_i.fourquark_Zs_q
    dZf = (Data_f.fourquark_sigmaJK_q)*(Data_f.fourquark_sigmaJK_q)
    dZi_inv = reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK_q, inv(Zi)])*\
              reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK_q, inv(Zi)])
    dSS = sqrt(dot(dZf, inv(Zi)) + dot(Zf, dZi_inv))
    Data_f.step_scale_sigma_q = dSS

   # (q, g)
    Zf = Data_f.fourquark_Zs_qg
    Zi = Data_i.fourquark_Zs_qg
    dZf = (Data_f.fourquark_sigmaJK_qg)*(Data_f.fourquark_sigmaJK_qg)
    dZi_inv = reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK_qg, inv(Zi)])*\
              reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK_qg, inv(Zi)])
    dSS = sqrt(dot(dZf, inv(Zi)) + dot(Zf, dZi_inv))
    Data_f.step_scale_sigma_qg = dSS

   # (q, q)
    Zf = Data_f.fourquark_Zs_qq
    Zi = Data_i.fourquark_Zs_qq
    dZf = (Data_f.fourquark_sigmaJK_qq)*(Data_f.fourquark_sigmaJK_qq)
    dZi_inv = reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK_qq, inv(Zi)])*\
              reduce(dot, [inv(Zi), Data_i.fourquark_sigmaJK_qq, inv(Zi)])
    dSS = sqrt(dot(dZf, inv(Zi)) + dot(Zf, dZi_inv))
    Data_f.step_scale_sigma_qq = dSS

def step_scale_JK(Data_f, Data_i):
    pass

def do_ss(denominator):
    "Return a function that calculates Z/denominator."
    return lambda d: step_scale(d, denominator)

def do_ssJK(denominator):
    "Return a function that calculates sigma_{Z/denominato}."
    return lambda d: step_scale_sigma(d, denominator)
