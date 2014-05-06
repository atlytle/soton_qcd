'''
01 May 2014
Changed the qslash tensor fourquark projectors to "chiral form".
This makes the Z-matrix diagonal but shouldn't affect the chiral-allowed
matrix elements.  This note is just verifying that.

I recompute the 32^3 ~3 GeV datapoint, based on IW_SUSY_analysis.py,
rather than recomputing and pickling the whole dataset,
and check vs. the old pickled dataset.
'''

import sys
sys.path.append('../')
import pickle
import numpy as np
from multiprocessing import Pool

import pyNPR as npr
import measurements as m
import fits

ar = np.array
np.set_printoptions(precision=4, suppress=True)

def load_IWf_Data(m, plist, twlist, gflist):
    return [npr.IWf_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]

# 24^3 x 64
#plist = [(-3,0,3,0)]
#twlist = [2.25]
#gflist_005 = [1000]

# 32^3 x 64
plist = [(-5,0,5,0)]
twlist = [-0.625]
gflist004 = [1700, 1740, 1780, 1820, 1860, 1900, 1940, 1980,
               2020, 2060, 2100, 2140, 2180, 2220, 2260]
gflist006 = [1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280,
               1320, 1360, 1400, 1440, 1480]
gflist008 = [1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 
               2120, 2160]

def calc_Zs(Data):
    Data.load()
    m.bilinear_Lambdas(Data)
    m.bilinear_LambdaJK(Data)
    m.fourquark_Zs(Data)
    m.fourquark_ZsJK(Data)
    Data.clear()
    return Data

def main():
    print 'Loading data..',
    data004 = load_IWf_Data(0.004, plist, twlist, gflist004)
    data006 = load_IWf_Data(0.006, plist, twlist, gflist006)
    data008 = load_IWf_Data(0.008, plist, twlist, gflist008)
    print 'done.'

    print 'Unpickling old result..',
    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/SUSY_BK'
    with open(pickle_root+'/IWf_chiral_pickle', 'r') as f:
        data0f_old = pickle.load(f)
    print 'done.'

    print 'Computing Zs..',
    pool = Pool()
    
    # Fine Zs.
    data004 = pool.map_async(calc_Zs, data004).get()
    data006 = pool.map_async(calc_Zs, data006).get()
    data008 = pool.map_async(calc_Zs, data008).get()

    pool.close()
    pool.join()
    print "done."

    print 'Chiral limit..',
    data0f = map(fits.line_fit_Data, data004, data006, data008)
    print 'done.'

    print "_________________32^3 Iwasaki NE data________________"
    print '(ap)^2:', data0f_old[-3].apSq, '   mu^2:', (data0f_old[-3].mu)**2, '\n'
    print 'Z^{-1} in (q, q) scheme:'
    print '--- Chiral result (old)---'
    print data0f_old[-3].Zinv_q, '\n+/-\n', data0f_old[-3].Zinv_q_sigmaJK
    print '--- Chiral result (new) ---'
    print '(ap)^2:', data0f[0].apSq, '   mu^2:', (data0f[0].mu)**2, '\n'
    print data0f[0].Zinv_q, '\n+/-\n', data0f[0].Zinv_q_sigmaJK
    
    return 0

if __name__ == "__main__":
    sys.exit(main())