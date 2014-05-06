import sys
sys.path.append('../')
import pickle
import numpy as np
from numpy.linalg import inv

import fits
import measurements as m

np.set_printoptions(precision=4, suppress=True)

# From SMOM-NDR_conversion.nb.
SMOMq_to_MSbar = np.array([  
[0.991113, 0, 0],
[0, 1.00084, 0.00507822],
[0, 0.00599621, 1.0293]])

def main():

    print "Un-pickling data...",
    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/SUSY_BK'
    with open(pickle_root+'/IWf_008_pickle', 'r') as f:
        data008 = pickle.load(f)
    with open(pickle_root+'/IWf_006_pickle', 'r') as f:
        data006 = pickle.load(f)
    with open(pickle_root+'/IWf_004_pickle', 'r') as f:
        data004 = pickle.load(f)
    with open(pickle_root+'/IWc_02_pickle', 'r') as f:
        data02 = pickle.load(f)
    with open(pickle_root+'/IWc_01_pickle', 'r') as f:
        data01 = pickle.load(f)
    with open(pickle_root+'/IWc_005_pickle', 'r') as f:
        data005 = pickle.load(f)
    with open(pickle_root+'/IWc_chiral_pickle', 'r') as f:
        data0c = pickle.load(f)
    with open(pickle_root+'/IWf_chiral_pickle', 'r') as f:
        data0f = pickle.load(f)
    
    print "complete."
    
    print "_________________24^3 Iwasaki NE data________________"
    for x in -3, -2:
        print '---------------------------------------------'
        print '(ap)^2:', data02[x].apSq, '   mu^2:', (data02[x].mu)**2, '\n'
        for d in data02[x], data01[x], data005[x]:
            print '--- am = ', d.m, '---\n'
            print 'Vq:', d.Vq, '\n'
            print 'Z^{-1} in (q, q) scheme:'
            print d.Zinv_q, '\n+/-\n', d.Zinv_q_sigmaJK
            print '\n\n'
        print '--- Chiral result ---'
        print data0c[x].Zinv_q, '\n+/-\n', data0c[x].Zinv_q_sigmaJK, '\n'
    print 'Set forbidden elements to zero and invert:\n'
    # Chiral Zs.
    mask_invert = lambda Zinv: inv(Zinv*m.chiral_mask)
    for x in -3, -2:
        print '(ap)^2:', data02[x].apSq, '   mu^2:', (data02[x].mu)**2
        Z = mask_invert(data0c[x].Zinv_q)
        print 'Z chiral:\n', Z, '\n'
        #JKsamples = map(mask_invert, data0c[x].Zinv_JK)
        #print 'sigma:\n', m.JKsigma(Z, JKsamples)
        #print data0c[x].Zinv_q_sigmaJK
    # Interpolation results.
    print 'Interpolation result at 3 GeV (linear interpolation):'
    p1 = (data0c[-3].mu, inv(data0c[-3].Zinv_q*m.chiral_mask))
    p2 = (data0c[-2].mu, inv(data0c[-2].Zinv_q*m.chiral_mask))
    a, b = fits.line_fit_2pt(p1, p2)  # y = a*x+b.
    SMOM_result = a*3+b
    print SMOM_result, '\n'
    print 'MSbar result at 3 GeV:'
    print np.dot(SMOMq_to_MSbar, SMOM_result[:3,:3]), '\n'
    
    print "_________________32^3 Iwasaki NE data________________"
    for x in -4, -3:
        print '---------------------------------------------'
        print '(ap)^2:', data008[x].apSq, '   mu^2:', (data008[x].mu)**2, '\n'
        for d in data008[x], data006[x], data004[x]:
            print '--- am = ', d.m, '---\n'
            print 'Vq:', d.Vq, '\n'
            print 'Z^{-1} in (q, q) scheme:'
            print d.Zinv_q, '\n+/-\n', d.Zinv_q_sigmaJK
            print '\n\n'
        print '--- Chiral result ---'
        print data0f[x].Zinv_q, '\n+/-\n', data0f[x].Zinv_q_sigmaJK

    print 'Set forbidden elements to zero and invert:\n'
    # Chiral Zs.
    for x in -4, -3:
        print '(ap)^2:', data008[x].apSq, '   mu^2:', (data008[x].mu)**2
        print 'Z chiral:\n', inv(data0f[x].Zinv_q*m.chiral_mask), '\n'
    # Interpolation results.
    print 'Interpolation result at 3 GeV (linear interpolation):'
    p1 = (data0f[-4].mu, inv(data0f[-4].Zinv_q*m.chiral_mask))
    p2 = (data0f[-3].mu, inv(data0f[-3].Zinv_q*m.chiral_mask))
    a, b = fits.line_fit_2pt(p1, p2)  # y = a*x+b.
    SMOM_result = a*3+b
    print SMOM_result, '\n'
    print 'MSbar result at 3 GeV:'
    print np.dot(SMOMq_to_MSbar, SMOM_result[:3,:3]), '\n'
    
if __name__ == "__main__":
    sys.exit(main())
