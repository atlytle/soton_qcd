import sys
import pickle
import itertools
import optparse
import numpy as np
import pylab as p
from multiprocessing import Pool

import fits, pole_fits
import pyNPR as npr
import measurements_exceptional as m

ar = np.array
np.set_printoptions(precision=5, suppress=True)

# Parameters
# 24^3 x 64

plist_IWc = [(-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), 
             (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-4,0,4,0)]
twlist_IWc = [0.1875, 0.375, 0.5625, 0.75, 1.125, 1.5, 1.6875, 2.25, 1.5]
gflist_03 = [980, 1140, 1300, 1460, 1620, 1780, 1940, 2100, 2260, 2420,
             2580, 2740, 2900, 3060]
gflist_02 = [1250, 1410, 1570, 1730, 1890, 2050, 2210, 2370, 2530, 2690,
             2850, 3010, 3170, 3330, 3490]
gflist_01 = [1460, 1620, 1780, 1940, 2100, 2260, 2420, 2580, 2740, 2900,
             3060, 3220, 3380, 3540, 3700, 3860, 4020, 4180, 4340, 4500,
             4660, 4820, 4980]
gflist_005 = [1000, 1160, 1320, 1480, 1640, 1800, 1960, 2120,
              2280, 2440, 2600, 2760, 2920, 3080, 3240, 3400, 3560, 3720,
              3880, 4040, 4200, 4360, 4520, 4680, 4840, 5000, 5160, 5320,
              5480, 5640, 5800, 5960, 6120, 6280, 6440, 6600]
              

def main():     

    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/'\
                  'IW_exceptional_24cube/'      
                  
    print "Un-pickling data..."
    
    with open(pickle_root+'IW_exceptional_24cube_03.pkl', 'r') as f:
        data_03 = pickle.load(f)
    with open(pickle_root+'IW_exceptional_24cube_02.pkl', 'r') as f:
        data_02 = pickle.load(f)
    with open(pickle_root+'IW_exceptional_24cube_01.pkl', 'r') as f:
        data_01 = pickle.load(f)  
    with open(pickle_root+'IW_exceptional_24cube_005.pkl', 'r') as f:
        data_005 = pickle.load(f)                     
    print "complete."
    
#    for x in data_03, data_02, data_01, data_005:
#        print x[-1].m
#        print x[-1].Zinv, '\n'
#        print x[-1].Zinv_sigmaJK
#        print ''
    
    print [x.mu for x in data_03]
    xarr = np.array([x[-1].m for x in data_03, data_02, data_01, data_005])
    chi2_ar = np.ones((5,5))
    Zinv_ar = np.ones((5,5))
    p0_ar = np.ones((5,5))
    p1_ar = np.ones((5,5))
    p2_ar = np.ones((5,5))
    for a in range(0,5):
        for b in range(0,5):
            # Need to wrap these as arrays.
            yarr = np.array([x[-1].Zinv[a,b] for x in data_03, data_02, data_01, data_005])
            earr = np.array([x[-1].Zinv_sigmaJK[a,b] for x in data_03, data_02, data_01, data_005])
            pfit, chi2 = pole_fits.single_double_nocst_fit(xarr, yarr, earr)
            p0_ar[a,b]=pfit[0]
            p1_ar[a,b]=pfit[1]
            p2_ar[a,b]=pfit[2]
            chi2_ar[a,b] = chi2
            Zinv_ar[a,b] = pfit[1]
    print chi2_ar
    print p0_ar/.005
    print p1_ar
    #print p2_ar
    #print np.linalg.inv(Zinv_ar)
        
              
if __name__ == "__main__":
    sys.exit(main())
