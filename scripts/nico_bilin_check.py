'''Quick check of Nicolas' code.  Email Apr 15, 2014.'''

import sys
sys.path.append('../')
import pickle
import numpy as np
from multiprocessing import Pool

import pyNPR as npr
import measurements as m

ar = np.array
np.set_printoptions(precision=2, suppress=True)

# 24^3 x 64
plist = [(-4,0,4,0)]
twlist = [1.5]
gflist_005 = [1000]

def load_IWc_Data(m, plist, twlist, gflist):
    return [npr.IWc_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]

def calc_Zs(Data):
    Data.load()
    m.bilinear_Lambdas(Data)
    #m.bilinear_LambdaJK(Data)
    Data.clear()
    return Data

def main():
    print 'Loading data...',
    data_005 = load_IWc_Data(0.005, plist, twlist, gflist_005)
    print 'done.'

    print 'Computing Zs...',
    pool = Pool()
    data_005 = pool.map_async(calc_Zs, data_005).get()
    pool.close()
    pool.join()
    print 'done.'
    
    d = data_005[0]
    print 'S -', d.Lambda[0]
    print 'V -', d.Lambda_V
    print 'A -', d.Lambda_A
    print 'P -', d.Lambda[15]
    print ''
    print 'Vq -', d.Vq
    print 'Aq -', d.Vq5

if __name__ == "__main__":
    sys.exit(main())
