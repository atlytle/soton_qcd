import sys, numpy
from numpy import dot
from numpy.linalg import inv
from multiprocessing import Pool

import fits
import pyNPR as npr
import measurements as m
from combined_analysis import propagate_errors

numpy.set_printoptions(precision=5, suppress=True)

# Parameters.
plist0042 = [(-3, 0, 3, 0), (-5, 0, 5, 0)]
twlist0042 = [0.0, 1.0]
gflist0042 = [704, 864, 1024, 1184, 1344, 1504, 1664, 1824]

plist001 = plist0042
twlist001 = twlist0042
gflist001 = [500, 564, 628, 692, 756, 820, 884, 948]
gflist001_b = [500, 516, 524, 532, 564, 580, 588, 596, 
               628, 644, 652, 660, 692, 708, 716, 724,
               756, 772, 788, 820, 836, 844, 852, 884, 
               900, 908, 916, 948, 964, 980, 1012]

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

    data0042 = load_DSDR_Data(.0042, plist0042, twlist0042, gflist0042)
    data001 = \
        load_DSDR_Data(.001, plist001[0:1], twlist001[0:1], gflist001_b) +\
        load_DSDR_Data(.001, plist001[-1:], twlist001[-1:], gflist001)

   
    # Compute results.
    pool = Pool()
    data001 = pool.map_async(calc_Zs, data001).get()
    data0042 = pool.map_async(calc_Zs, data0042).get()
    pool.close()
    pool.join()

    data0 = map(fits.line_fit_2ptData, data001, data0042)
    
    M = numpy.array([[1.,0,0],[0,2.,0],[0,0,-4.]])  # Change to delta S=1 basis.

    print [d.mu for d in data0]
    
    d1 = data0[0]
    print 'DSDR', d1.mu, 'GeV \n'
    print d1.thing_chi, '\n'
    print d1.thing_chi_sigma,'\n'
    
    print 'q-scheme:'
    print d1.thing_q_chi, '\n'
    print d1.thing_q_chi_sigma,'\n'
    #print dot(dot(M,thing), inv(M))

    return 0

if __name__ == "__main__":
    sys.exit(main())
