import sys
import pickle
from multiprocessing import Pool

import pyNPR as npr
import measurements_exceptional as m

# Parameters
# 32^3 x 64
plistIWf = [(-3, 0, 3, 0), (-4, 0, 4, 0), (-4, 0, 4, 0),
            (-5, 0, 5, 0), (-5, 0, 5, 0)]
twlistIWf = [0.25, -0.75, .375, -0.625, 0.375]
gflist004 = [1700, 1740, 1780, 1820, 1860, 1900, 1940, 1980,
             2020, 2060, 2100, 2140, 2180, 2220, 2260]
gflist006 = [1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280,
             1320, 1360, 1400, 1440, 1480]
gflist008 = [1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080,
             2120, 2160]

def load_IWf_Data(m, plist, twlist, gflist):
    return [npr.IWf_Exceptional_Data(m, plist[i], twlist[i], gflist)
            for i in range(len(plist))]

def calc_Zs(Data):
    Data.load()
    m.bilinear_Lambdas(Data)
    m.bilinear_LambdaJK(Data)
    m.fourquark_Zs(Data)
    return Data

def main():
    compute = True
    dump = True
    load = False
    if compute:
        data004 = load_IWf_Data(.004, plistIWf, twlistIWf, gflist004)
        data006 = load_IWf_Data(.006, plistIWf, twlistIWf, gflist006)
        data008 = load_IWf_Data(.008, plistIWf, twlistIWf, gflist008)

        #print [d.mu for d in data004]

        pool = Pool()

        data004 = pool.map_async(calc_Zs, data004).get()
        data006 = pool.map_async(calc_Zs, data006).get()
        data008 = pool.map_async(calc_Zs, data008).get()

        pool.close()
        pool.join()

    if compute and dump:
        print "Pickling data..."
        root = '/Users/atlytle/Dropbox/pycode/soton/pickle/IW_exceptional'
        with open(root+'/IWf_exceptional_004.pkl', 'w') as f:
            pickle.dump(data004, f)

    '''
    print "Lambda_V:", [d.Lambda_V for d in data004]
    print "Lambda_A:", [d.Lambda_A for d in data004]
    print "Lambda_VpA_sigmaJK:", [d.Lambda_VpA_sigmaJK for d in data004]
    print "Fourquark Lambdas:"
    for d in data004:
        print d.Zinv
    '''
    # Naive chiral limit
    # Naive pole subtraction
    # Less-naive chiral limit
    # Plots.
    return 0

if __name__ == "__main__":
    sys.exit(main())
