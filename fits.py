import numpy as np
import matplotlib.pyplot as plot
from scipy import interpolate
from numpy import dot
from numpy.linalg import inv

from pyNPR import Data
from measurements import F_gg, F_qq, chiral_mask
from combined_analysis import propagate_errors

def line_fit_2pt(p1, p2):
    '''Simple test function.'''
    x1, y1 = p1
    x2, y2 = p2
    m = (y2-y1)/(x2-x1)
    b = y1 - m*x1
    return m, b

def line_fit_2ptData(Data1, Data2):
    '''Chiral extrapolate fourquark Z-matrix on two Data points.'''
    mres = Data1.mres
    result = Data1.__class__(-mres, Data1.p, Data1.tw, None)
    
    # Chiral extrap of Z^chi Lambda^f
    p1 = (Data1.m + mres, Data1.thing, Data1.thing_sigmaJK)
    p2 = (Data2.m + mres, Data2.thing, Data2.thing_sigmaJK)
    result.thing_chi, result.thing_chi_sigma = line_fit([p1,p2])
    
    # Chiral extrap of Z^chi Lambda_q^f
    p1 = (Data1.m + mres, Data1.thing_q, Data1.thing_q_sigmaJK)
    p2 = (Data2.m + mres, Data2.thing_q, Data2.thing_q_sigmaJK)
    result.thing_q_chi, result.thing_q_chi_sigma = line_fit([p1,p2])
    
    # Chiral extrap of Lambda
    result.fourquark_Lambda = {}
    p1 = (Data1.m + mres, Data1.fourquark_Lambda, Data1.Lambda_sigmaJK)
    p2 = (Data2.m + mres, Data2.fourquark_Lambda, Data2.Lambda_sigmaJK)
    result.fourquark_Lambda, result.Lambda_sigmaJK = line_fit([p1,p2])
    result.Zinv = dot(result.fourquark_Lambda, inv(F_gg))  # need JK
    result.Zinv_sigma = propagate_errors([result.fourquark_Lambda, inv(F_gg)],
                                         [result.Lambda_sigmaJK, np.zeros((5,5))])
    result.Zinv_chi = dot(result.fourquark_Lambda*chiral_mask, inv(F_gg))
    result.Z_chi = inv(result.Zinv_chi)
    result.thing = dot(result.Z_chi, result.Zinv)

    # Chiral extrap of Zs.
    result.fourquark_Zs = {} #argument for putting these in class def?
    result.fourquark_sigmaJK = {}
    for s in 'gg', 'gq', 'qg', 'qq':
        p1 = (Data1.m + mres, Data1.fourquark_Zs[s], Data1.fourquark_sigmaJK[s])
        p2 = (Data2.m + mres, Data2.fourquark_Zs[s], Data2.fourquark_sigmaJK[s])
        result.fourquark_Zs[s], result.fourquark_sigmaJK[s] = line_fit([p1, p2])
        
        # HACK to remove nans that occur from zeroing elements in data
        areNans = np.isnan(result.fourquark_Zs[s])
        result.fourquark_Zs[s][areNans] = 0.
        areNans = np.isnan(result.fourquark_sigmaJK[s])
        result.fourquark_sigmaJK[s][areNans] = 0.
 
    return result

def line_fit_Data(*Dat):
    '''Chiral extrapolate fourquark Z-matrices.'''

    for d in Dat:
        assert isinstance(d, Data)

    # Initialize result.
    d0 = Dat[0]
    mres = d0.mres
    result = d0.__class__(-mres, d0.p, d0.tw, None)
    
    # Chiral extrap of Z^chi Lambda^f
    points = [(d.m + mres, d.thing, d.thing_sigmaJK) for d in Dat]
    result.thing_chi, result.thing_chi_sigma = line_fit(points)
    
    # Chiral extrap of Z^chi Lambda_q^f
    points = [(d.m + mres, d.thing_q, d.thing_q_sigmaJK) for d in Dat]
    result.thing_q_chi, result.thing_q_chi_sigma = line_fit(points)
    
    # Chiral extrap of Lambda.
    result.fourquark_Lambda = {}
    points = [(d.m + mres, d.fourquark_Lambda, d.Lambda_sigmaJK)
              for d in Dat]
    result.fourquark_Lambda, result.Lambda_sigmaJK = line_fit(points)
    result.Zinv = dot(result.fourquark_Lambda, inv(F_gg))  # need JK
    result.Zinv_chi = dot(result.fourquark_Lambda*chiral_mask, inv(F_gg))
    result.Z_chi = inv(result.Zinv_chi)
    result.thing = dot(result.Z_chi, result.Zinv)
    
    # Chiral extrap of Lambda_q.
    result.fourquark_Lambda_q = {}
    points = [(d.m + mres, d.fourquark_Lambda_q, d.Lambda_sigmaJK_q)
              for d in Dat]
    result.fourquark_Lambda_q, result.Lambda_sigmaJK_q = line_fit(points)
    result.Zinv_q = dot(result.fourquark_Lambda_q, inv(F_qq))
     
    # Chiral extrap of Zs.    
    result.fourquark_Zs = {} #argument for putting these in class def?
    result.fourquark_sigmaJK = {}
    for s in 'gg', 'gq', 'qg', 'qq':
        points = [(d.m + mres, d.fourquark_Zs[s], d.fourquark_sigmaJK[s]) 
                  for d in Dat]
        result.fourquark_Zs[s], result.fourquark_sigmaJK[s] = line_fit(points)
        
        # HACK to remove nans that occur from zeroing elements in data
        areNans = np.isnan(result.fourquark_Zs[s])
        result.fourquark_Zs[s][areNans] = 0.
        areNans = np.isnan(result.fourquark_sigmaJK[s])
        result.fourquark_sigmaJK[s][areNans] = 0.

    return result

def line_fit(data):
    '''
    Fits data in (x, y, sig) format to a straight line.
    Expressions used are from Numerical Recipes.
    '''
    dl = data
    S = lambda dl: sum([1/(d[2]*d[2]) for d in dl])
    Sx = lambda dl: sum([d[0]/(d[2]*d[2]) for d in dl])
    Sy = lambda dl: sum([d[1]/(d[2]*d[2]) for d in dl])
    Sxx = lambda dl: sum([d[0]*d[0]/(d[2]*d[2]) for d in dl])
    Sxy = lambda dl: sum([d[0]*d[1]/(d[2]*d[2]) for d in dl])
    delta = lambda dl: S(dl)*Sxx(dl) - Sx(dl)*Sx(dl)

    a = lambda dl: (Sxx(dl)*Sy(dl) - Sx(dl)*Sxy(dl))/delta(dl)
    b = lambda dl: (S(dl)*Sxy(dl) - Sx(dl)*Sy(dl))/delta(dl)
    sigAsq = lambda dl: Sxx(dl)/delta(dl)
    sigBsq = lambda dl: S(dl)/delta(dl)
    Cov = lambda dl: -Sx(dl)/delta(dl)
    r = lambda dl: -Sx(dl)/np.sqrt(S(dl)*Sxx(dl))
    return a(dl), np.sqrt(sigAsq(dl))

def line_fit_Lambda(Data1, Data2):
    "Wanted to test extrapolation in Lambda instead of Z."
    mres = .001853
    result = Data(-mres, Data1.p, Data1.tw, None)
    p1 = (Data1.m + mres, Data1.fourquark_Lambda/(Data1.Lambda_VpA)**2, 
                          Data1.Lambda_sigmaJK) # bit of a fudge, only 2 pts
    p2 = (Data2.m + mres, Data2.fourquark_Lambda/(Data2.Lambda_VpA)**2, 
                          Data2.Lambda_sigmaJK)
    result.fourquark_Lambda = line_fit([p1, p2])[0] # up to VpAs
    result.Lambda_sigmaJK = line_fit([p1, p2])[1]
    result.fourquark_Zs = np.dot(F_gg, np.linalg.inv(result.fourquark_Lambda))
    result.fourquark_sigmaJK = np.zeros((5,5))
    return result

def spline_interpolate(Data_list):
   pass 

