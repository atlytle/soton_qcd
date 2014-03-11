from pylab import *
from scipy import *
import numpy
from scipy import optimize
numpy.set_printoptions(precision=3)

from pyNPR import Data
from measurements import F_gg, JKsigma
from fits import line_fit2

def test_fit(x, y, err):
    fitfunc = lambda p, x: p[0]/(x*x) + p[1]/x +  p[2]
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err

    p0 = [.5, .5, .5]
    p1, success = optimize.leastsq(errfunc, p0[:], args=(x,y,err))

    print "Input parameters:", p
    print "Fit parameters:  ", p1

def single_double_pole_fit(xarr, yarr, earr):
    '''Fit data to form A/x^2 + B/x + C + D*x'''
    fitfunc = lambda p, x: p[0]/(x*x) + p[1]/x + p[2] + p[3]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err
    p0 = [1., 1., 1., 1.]  # Initial guess.

    p1, success = optimize.leastsq(errfunc, p0, args=(xarr,yarr,earr),
                                   full_output=0)
    # Compute chi^2 of the fit.
    trialfunc = lambda x: fitfunc(p1,x)
    chi_sq= chisq(xarr, yarr, earr, trialfunc)
    
    return p1, chi_sq

def double_pole_fit(xarr, yarr, earr):
    '''Fit data to form A/x^2 + B + C*x'''
    fitfunc = lambda p, x: p[0]/(x*x) + p[1] + p[2]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err
    p0 = [1., 1., 1.]  # Initial guess.

    p1, success = optimize.leastsq(errfunc, p0, args=(xarr,yarr,earr),
                                   full_output=0)
                                   
    # Compute chi^2 of the fit.
    trialfunc = lambda x: fitfunc(p1,x)
    chi_sq= chisq(xarr, yarr, earr, trialfunc)
    
    return p1, chi_sq
    
def chisq(xarr, yarr, earr, trialfunc):
    return sum([((y-trialfunc(x))/err)**2 
                for x, y, err in zip(xarr, yarr, earr)])
                
def single_double_nocst_fit(xarr, yarr, earr):
    '''Fit data to form A/x^2 + B/x + C*x'''
    fitfunc = lambda p, x: p[0]/(x*x) + p[1]/x + p[2]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err
    p0 = [1., 1., 1.]  # Initial guess.

    p1, success = optimize.leastsq(errfunc, p0, args=(xarr,yarr,earr),
                                   full_output=0)
                                   
    # Compute chi^2 of the fit.
    trialfunc = lambda x: fitfunc(p1,x)
    chi_sq= chisq(xarr, yarr, earr, trialfunc)
    
    return p1, chi_sq
    
def single_double_nolin_fit(xarr, yarr, earr):
    '''Fit data to form A/x^2 + B/x + C'''
    fitfunc = lambda p, x: p[0]/(x*x) + p[1]/x + p[2]
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err
    p0 = [1., 1., 1.]  # Initial guess.

    p1, success = optimize.leastsq(errfunc, p0, args=(xarr,yarr,earr),
                                   full_output=0)
                                   
    # Compute chi^2 of the fit.
    trialfunc = lambda x: fitfunc(p1,x)
    chi_sq= chisq(xarr, yarr, earr, trialfunc)
    
    return p1, chi_sq
    
    
def single_pole_fit(xarr, yarr, earr):
    '''Fit data to form A/x + B +C*x'''
    fitfunc = lambda p, x: p[0]/x + p[1] + p[2]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err
    p0 = [1., 1., 1.]  # Initial guess.

    p1, success = optimize.leastsq(errfunc, p0, args=(xarr,yarr,earr),
                                   full_output=0)
                                   
    # Compute chi^2 of the fit.
    trialfunc = lambda x: fitfunc(p1,x)
    chi_sq= chisq(xarr, yarr, earr, trialfunc)
    
    return p1, chi_sq

def linear_fit(xarr, yarr, earr):
    '''Fit data to form A + B*x'''
    fitfunc = lambda p, x: p[0] + p[1]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err
    p0 = [1., 1.]  # Initial guess.

    p1, success = optimize.leastsq(errfunc, p0, args=(xarr,yarr,earr),
                                   full_output=0)
    return p1

def pole_subtract(m1, d1, m2, d2):
    '''Subtract 1/m term in data.'''
    return (m1*d1 - m2*d2)/(m1-m2)
    
def pole_subtract_Data(Data1, Data2):
    for d in Data1, Data2:
        assert isinstance(d, Data)
        
    # Initialize result. 
        # Note here* the chiral limit maintained at -mres.
        # This is because our mass parameters are defined as input masses.
    d1, d2 = Data1, Data2
    mres = d1.mres  # = d2.mres
    m1, m2 = (d1.m + mres, d2.m + mres)
    VpA1, VpA2 = d1.Lambda_VpA, d2.Lambda_VpA
    result = d1.__class__(m1 + m2 - mres, d1.p, d1.tw, None)  # *
    # Note this Lambda has VpA included in definition.
    result.fourquark_Lambda = pole_subtract(m1, d1.fourquark_Lambda/(VpA1*VpA1),
                                            m2, d2.fourquark_Lambda/(VpA2*VpA2))
    result.fourquark_Zs = dot(F_gg, inv(result.fourquark_Lambda))
    # Uncertainty - MUST FIX! not quite sure how yet...
    s1 = m1*(d1.fourquark_Lambda_sigmaJK**2)/(VpA1*VpA1)
    s2 = m2*d2.fourquark_Lambda_sigmaJK**2/(VpA2*VpA2)
    result.fourquark_Lambda_sigmaJK = (s1 + s2)/(m1 - m2)
    result.fourquark_sigmaJK = np.ones((5,5))

    
    return result
    
def pole_subtract_Data2(*Dat):
    '''Remove 1/m dependence in data pts by fitting m*Zinv to a straight line.'''
    for d in Dat:
        assert isinstance(d, Data)
    Dat = list(Dat) 
    # Multiply by am and find intercept. 
    mres = Dat[0].mres  
    points = [(mres + d.m, (mres + d.m)*d.Zinv, (mres + d.m)*d.Zinv_sigmaJK)
             for d in Dat]
    fit = line_fit2(points)
    for d in Dat:
        d.polefit_params = fit  # Save these parameters.
        d.Zinv_sub = d.Zinv - fit.a/(d.m + mres)
        d.Zinv_sub_sigma = sqrt(d.Zinv_sigmaJK**2)# + (fit.sig_a/(d.m+mres))**2)
        
    # Jackknife values.
    # This proceeds in a few steps. First we calculate the fit parameter 'a'
    # many times by removing individual configs.  Then we enlarge the
    # "jackknife space" of each data point to include the effect on the 
    # subtracted value coming from the other mass points.
    
    aJK = []
    for dummy in range(len(Dat)):
        d, ds = Dat[0], Dat[1:]  # Split off first element.
        d.Zinv_subJK = []
        for jk in d.Zinv_JK:
            # We do not recompute d.Zinv_sigmaJK which would require some
            # insane double jackknife.
            points = [(mres + x.m, (mres + x.m)*x.Zinv, 
                     (mres + x.m)*x.Zinv_sigmaJK) for x in ds] +\
                     [(mres + d.m, (mres + d.m)*jk, (mres + d.m)*d.Zinv_sigmaJK)]
            fit = line_fit2(points)
            aJK += [fit.a]
        ds.append(d) # Return element to end.
        Dat = ds
    
    # Initialize Zinv_subJK.  The central value d.Zinv is used when considering
    # the effect from other mass points.  In the following loop we correct
    # the values in the same mass point.
    
    ct = 0
    Zinv_subJK = []
    for d in Dat:
        d.Zinv_JKexpand = [d.Zinv]*len(aJK)  #  Init. expanded 'regular' JK.
        for x in range(len(d.Zinv_JK)):
            Zinv_subJK += [d.Zinv - aJK[ct]/(mres+d.m)]
            d.Zinv_JKexpand[ct] = d.Zinv_JK[x]  # Fill JK values in correct spots.
            ct +=1           
    assert ct == len(aJK)
    
    ct = 0
    for dummy in range(len(Dat)):
        d, ds = Dat[0], Dat[1:]
        d.Zinv_subJK = Zinv_subJK  # Initialize.
        for jk in d.Zinv_JK:
            d.Zinv_subJK[ct] = jk - aJK[ct]/(mres+d.m)  # Corrects wrong values.
            ct += 1
        d.Zinv_sub_sigma2 = JKsigma(d.Zinv_subJK, d.Zinv_sub)
        ds.append(d)
        Dat = ds
    assert ct == len(aJK)
    

if __name__ == "__main__":
    num = 40
    x = linspace(.5, 4, num)
    p = array([rand(),rand(),rand()])
    y = p[0]/(x*x) + p[1]/x + p[2] + .1*(rand(num)-.5)
    err = 1*ones(num)
    
    test_fit(x, y, err)
    print single_double_pole_fit(x, y, err)
    print double_pole_fit(x, y, err)
    print linear_fit(x, y, err)
