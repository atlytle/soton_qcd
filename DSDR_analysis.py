import sys
import pickle
import numpy as np
from multiprocessing import Pool

import domain_wall as dw
import pyNPR as npr
import measurements as m
import output as out
import step_scaling as ss
import fits

# Parameters.
plist0042 = [(-3, 0, 3, 0), (-4, 0, 4, 0), (-4, 0, 4, 0), (-4, 0, 4, 0),
             (-4, 0, 4, 0), (-5, 0, 5, 0), (-5, 0, 5, 0), (-5, 0, 5, 0), 
             (-5, 0, 5, 0)]
twlist0042 = [0.0, -0.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5, 1.0]
gflist0042 = [704, 864, 1024, 1184, 1344, 1504, 1664, 1824]

plist001 = plist0042
twlist001 = twlist0042
gflist001 = [500, 564, 628, 692, 756, 820, 884, 948]
gflist001_ = [500, 516, 532, 564, 580, 596, 628, 644,
              660, 692, 708, 724, 756, 772, 788, 820,
              836, 852, 884, 916, 948, 980, 1012]

def load_DSDR_Data(m, plist, twlist, gflist):
    return [npr.DSDR_Data(m, plist[i], twlist[i], gflist)
            for i in range(len(plist))]

def calc_Zs(Data):
    Data.load()
    m.bilinear_Lambdas(Data)
    m.bilinear_LambdaJK(Data)
    m.fourquark_Zs(Data)
    m.fourquark_ZsJK(Data)
    Data.clear()
    return Data

def main():
    
    compute = True
    dump = True
    load = False

    if compute:
        # Load data.
        data0042 = load_DSDR_Data(.0042, plist0042, twlist0042, gflist0042)
        data001 = \
            load_DSDR_Data(.001, plist001[0:1], twlist001[0:1], gflist001_) +\
            load_DSDR_Data(.001, plist001[1:2], twlist001[1:2], gflist001) +\
            load_DSDR_Data(.001, plist001[2:3], twlist001[2:3], gflist001_) +\
            load_DSDR_Data(.001, plist001[3:], twlist001[3:], gflist001)

        #data001 = load_DSDR_Data(.001, plist001, twlist001, gflist001)

        # Compute results.
        pool = Pool()
        data001 = pool.map_async(calc_Zs, data001).get()
        data0042 = pool.map_async(calc_Zs, data0042).get()
        pool.close()
        pool.join()

        data0 = map(fits.line_fit_2ptData, data001, data0042)

    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle'
    kosher = pickle_root + '/DSDR_chiral_pickle'

    if dump:
        with open(kosher, 'w') as f:
            pickle.dump(data0, f)

    if load:
        with open(kosher, 'r') as f:
            data0 = pickle.load(f)
    
    # Output results.
    #root = '/Users/atlytle/Dropbox/TeX_docs/AuxDet_NPR/fourFermi/plots'
    #out.write_Zs(data0042, root + '/Z_am0042_non-exceptional_gamma_new.dat')
    #out.write_Zs(data001,  root + '/Z_am001_non-exceptional_gamma_new.dat')
    #out.write_Zs(data0, root + '/Z_chiral_non-exceptional_gamma.dat')
    
    #root = '/Users/atlytle/Dropbox/TeX_docs/soton/Kpipi/NPR/'
    #out.write_Zs_TeX_2([data0[2]], root + 'Z_chiral_1.5')
    #out.write_Zs_TeX_2([data0[0]], root + 'Z_chiral_1.1')
    return 0

if __name__ == "__main__":
    sys.exit(main())
