import sys
import pickle
import numpy as np
from numpy.linalg import inv

import fits, pole_fits
from measurements import chiral_mask
import measurements_exceptional as m

np.set_printoptions(precision=5, suppress=True)

MOM_to_MSbar = np.array([
[1.01716, 0, 0, 0, 0],
[0, 0.977953, -0.13228, 0, 0],
[0, 0.005993, 1.21233, 0, 0],
[0, 0, 0, 1.11023, 0.016719],
[0, 0, 0, 0.0631787, 1.05252]])

def main():
    print "Loading 24^3 IW exceptional data...",
    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/'\
                  'IW_exceptional_24cube/'
                  
    with open(pickle_root +'IW_exceptional_24cube_03.pkl', 'r') as f:
        data_03 = pickle.load(f)
    with open(pickle_root +'IW_exceptional_24cube_02.pkl', 'r') as f:
        data_02 = pickle.load(f)
    with open(pickle_root +'IW_exceptional_24cube_01.pkl', 'r') as f:
        data_01 = pickle.load(f)  
    with open(pickle_root +'IW_exceptional_24cube_005.pkl', 'r') as f:
        data_005 = pickle.load(f)  
    print "complete."
        
    print "Loading 32^3 IW exceptional data...",
    root = '/Users/atlytle/Dropbox/pycode/soton/pickle/IW_exceptional'        
    with open(root+'/IWf_exceptional_004.pkl', 'r') as f:
        data004 = pickle.load(f)
    with open(root+'/IWf_exceptional_006.pkl', 'r') as f:
        data006 = pickle.load(f)
    with open(root+'/IWf_exceptional_008.pkl', 'r') as f:
        data008 = pickle.load(f)
    print "complete."
    
    print "Performing pole subtractions...",
    map(pole_fits.pole_subtract_Data2, data_005, data_01, data_02)
    map(pole_fits.pole_subtract_Data2, data004, data006, data008)
    print "complete."
    
    data_0c = map(fits.line_fit_Data_e, data_005, data_01, data_02)
    data_0 = map(fits.line_fit_Data_e, data004, data006, data008)
        
    print "______________24^3 IW Exceptional data________________"
    for x in -2, -1,:
        print '-------------------------------------------'
        print '(ap)^2:', data_02[x].apSq, '  mu^2:', (data_02[x].mu)**2, '\n'
        for d in data_02[x], data_01[x], data_005[x]:
            print '--- am = ', d.m, '---\n'
            print 'Lambda_A:', d.Lambda_A, '  Lambda_V:', d.Lambda_V, '\n'
            print 'Z^{-1} in RI scheme (naive):\n'
            print d.Zinv, '\n+/-\n', d.Zinv_sigmaJK, '\n'
            print 'Z^{-1} in RI scheme after subtraction:\n'
            print d.Zinv_sub, '\n+/-\n', d.Zinv_sub_sigma, '\n'
    print 'Set forbidden elements to zero and invert:\n'
    # Chiral Zs.
    for x in -2, -1:
        print '(ap)^2:', data_02[x].apSq, '  mu^2:', (data_02[x].mu)**2
        print 'Z chiral:\n', inv(data_0c[x].Zinv_sub*chiral_mask), '\n'
    # Interpolation results.
    print 'Interpolation result at 3 GeV (linear interpolation):'
    p1 = (data_0c[-2].mu, inv(data_0c[-2].Zinv_sub*chiral_mask))
    p2 = (data_0c[-1].mu, inv(data_0c[-1].Zinv_sub*chiral_mask))
    a, b = fits.line_fit_2pt(p1, p2)  # y = a*x+b.
    MOM_result = a*3+b
    print MOM_result, '\n'
    print 'MSbar result at 3 GeV:'
    print np.dot(MOM_to_MSbar, MOM_result), '\n'
    

    
#    print "______________32^3 IW Exceptional data________________" 
#    for x in -3, -2:
#        print '-------------------------------------------'
#        print '(ap)^2:', data008[x].apSq, '  mu^2:', (data008[x].mu)**2, '\n'
#        for d in data008[x], data006[x], data004[x]:
#            print '--- am = ', d.m, '---\n'
#            print 'Lambda_A:', d.Lambda_A, '  Lambda_V:', d.Lambda_V, '\n'
#            print 'Z^{-1} in RI scheme (naive):\n'
#            print d.Zinv, '\n+/-\n', d.Zinv_sigmaJK, '\n'
#            print 'Z^{-1} in RI scheme after subtraction:\n'
#            print d.Zinv_sub, '\n+/-\n', d.Zinv_sub_sigma, '\n' 
#        
#        print 'Pole coefficients as determined from linear fit method.'
#        print data004[x].polefit_params.a, ' +/-', data004[x].polefit_params.sig_a, '\n\n'
#        for d in data_0[x],:
#            print '--- Chiral limit ---'
#            print d.Zinv_sub, '\n'
#            
#    for i in range(5):
#        for j in range(5):
#            print 'i =', i+1, ' j=', j+1
#            for d in data008[-2], data006[-2], data004[-2], data_0[-2]:
#                print d.Zinv_sub[i,j], ' +/-', d.Zinv_sub_sigma[i,j] 
#            print ''
#            
##            print inv(d.Zinv_sub*chiral_mask), '\n'
##    # Cobble together bits of Zinv and return Z for each m.
##    for x in data004, data006, data008:
##        map(m.Zsub, x)
##    # Linear extrapolate results.
##    data_sub = map(fits.line_fit_Data_e, data004, data006, data008)
##    for x in -2,:
##        print "Subtracted Z in MOM scheme:"
##        print '(ap)^2:', data008[x].apSq, '  mu^2:', (data008[x].mu)**2, '\n'
##        print data_sub[x].Zsub, '\n+/-'
##        print data_sub[x].Zsub_sigmaJK, '\n'
##        print "Subtracted Z in MSbar scheme:"
##        print np.dot(MOM_to_MSbar, data_sub[x].Zsub), '\n'
##    print 'Interpolation result at 3 GeV (linear interpolation):'
##    p1 = (data_sub[-3].mu, data_sub[-3].Zsub)
##    p2 = (data_sub[-2].mu, data_sub[-2].Zsub)
##    a, b = fits.line_fit_2pt(p1, p2)  # y = a*x + b.
##    MOM_result = a*3+b
##    print MOM_result, '\n+/-'
##    print data_sub[-3].Zsub_sigmaJK, '\n' # Conservative error.
#    
#    print 'Set forbidden elements to zero and invert:\n'
#    # Chiral Zs.
#    for x in -3, -2:
#        print '(ap)^2:', data008[x].apSq, '  mu^2:', (data008[x].mu)**2
#        print 'Z chiral:\n', inv(data_0[x].Zinv_sub*chiral_mask), '\n'
#    # Interpolation results.
#    print 'Interpolation result at 3 GeV (linear interpolation):'
#    p1 = (data_0[-3].mu, inv(data_0[-3].Zinv_sub*chiral_mask))
#    p2 = (data_0[-2].mu, inv(data_0[-2].Zinv_sub*chiral_mask))
#    a, b = fits.line_fit_2pt(p1, p2)  # y = a*x+b.
#    MOM_result = a*3+b
#    print MOM_result, '\n'
#    print 'MSbar result at 3 GeV:'
#    print np.dot(MOM_to_MSbar, MOM_result), '\n'
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
