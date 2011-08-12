import numpy as np
import matplotlib.pyplot as plot
from scipy import interpolate

from pyNPR import Data #, F_gg

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

    # (g, g) - scheme
    p1 = (Data1.m + mres, Data1.fourquark_Zs, Data1.fourquark_sigmaJK)
    p2 = (Data2.m + mres, Data2.fourquark_Zs, Data2.fourquark_sigmaJK)
    result.fourquark_Zs = line_fit([p1, p2])[0]
    result.fourquark_sigmaJK = line_fit([p1, p2])[1] # inefficient

    # (g, q) - scheme
    p1 = (Data1.m + mres, Data1.fourquark_Zs_q, Data1.fourquark_sigmaJK_q)
    p2 = (Data2.m + mres, Data2.fourquark_Zs_q, Data2.fourquark_sigmaJK_q)
    result.fourquark_Zs_q = line_fit([p1, p2])[0]
    result.fourquark_sigmaJK_q = line_fit([p1, p2])[1] # inefficient
    
    # (q, g) - scheme
    p1 = (Data1.m + mres, Data1.fourquark_Zs_qg, Data1.fourquark_sigmaJK_qg)
    p2 = (Data2.m + mres, Data2.fourquark_Zs_qg, Data2.fourquark_sigmaJK_qg)
    result.fourquark_Zs_qg = line_fit([p1, p2])[0]
    result.fourquark_sigmaJK_qg = line_fit([p1, p2])[1] # inefficient

    # (q, q) - scheme
    p1 = (Data1.m + mres, Data1.fourquark_Zs_qq, Data1.fourquark_sigmaJK_qq)
    p2 = (Data2.m + mres, Data2.fourquark_Zs_qq, Data2.fourquark_sigmaJK_qq)
    result.fourquark_Zs_qq = line_fit([p1, p2])[0]
    result.fourquark_sigmaJK_qq = line_fit([p1, p2])[1] # inefficient

    return result

def line_fit_Data(*Dat):
    '''Chiral extrapolate fourquark Z-matrices.'''

    for d in Dat:
        assert isinstance(d, Data)

    # Initialize result.
    d0 = Dat[0]
    mres = d0.mres
    result = d0.__class__(-mres, d0.p, d0.tw, None)
    
    # (g, g) - scheme
    points = [(d.m + mres, d.fourquark_Zs, d.fourquark_sigmaJK)
              for d in Dat]
    result.fourquark_Zs, result.fourquark_sigmaJK = line_fit(points)
    
    # (g, q) - scheme
    points = [(d.m + mres, d.fourquark_Zs_q, d.fourquark_sigmaJK_q)
              for d in Dat]
    result.fourquark_Zs_q, result.fourquark_sigmaJK_q = line_fit(points)

    # (q, g) - scheme
    points = [(d.m + mres, d.fourquark_Zs_qg, d.fourquark_sigmaJK_qg)
              for d in Dat]
    result.fourquark_Zs_qg, result.fourquark_sigmaJK_qg = line_fit(points)

    # (q, q) - scheme
    points = [(d.m + mres, d.fourquark_Zs_qq, d.fourquark_sigmaJK_qq)
              for d in Dat]
    result.fourquark_Zs_qq, result.fourquark_sigmaJK_qq = line_fit(points)

    return result

def line_fit(data):
    '''
    Fits data in (x, y, sig) format to a straight line.
    Expressions used are from Numerical Recipes.
    '''
    #result = Data(0.0, data[0].p, data[0].tw, None)
    mres = .001853
    dl = data
    #dl = [(d.m + mres, d.fourquark_Zs, d.fourquark_sigmaJK) for d in data]
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

#    print S(dl)
#    print Sx(dl)
#    print Sy(dl)
#    print Sxx(dl)
#    print Sxy(dl)
#    print ''
#    print a(dl)
#    print b(dl)
#    print sigAsq(dl)
#    print sigBsq(dl)
#    print Cov(dl)
#    print r(dl)

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

