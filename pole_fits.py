from pylab import *
from scipy import *
import numpy
from scipy import optimize
numpy.set_printoptions(precision=3)

fitfunc = lambda p, x: p[0]/(x*x) + p[1]/x +  p[2]
errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err

num = 40
x = linspace(.5, 4, num)
p = array([rand(),rand(),rand()])
err = 1*ones(num)
y = p[0]/(x*x) + p[1]/x + p[2] + .1*(rand(num)-.5)

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
    return p1

def double_pole_fit(xarr, yarr, earr):
    '''Fit data to form A/x^2 + B + C*x'''
    fitfunc = lambda p, x: p[0]/(x*x) + p[1] + p[2]*x
    errfunc = lambda p, x, y, err: (y - fitfunc(p,x))/err
    p0 = [1., 1., 1.]  # Initial guess.

    p1, success = optimize.leastsq(errfunc, p0, args=(xarr,yarr,earr),
                                   full_output=0)
    return p1

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


print single_double_pole_fit(x, y, err)
print double_pole_fit(x, y, err)
print linear_fit(x, y, err)
