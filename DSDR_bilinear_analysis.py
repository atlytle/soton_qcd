import sys
import numpy as np
import pylab as p
from multiprocessing import Pool

import fits
import domain_wall as dw
import pyNPR as npr
import measurements as m

np.set_printoptions(precision=10)

# Parameters.
plist0042 = [(-3, 0, 3, 0), (-4, 0, 4, 0), (-4, 0, 4, 0), (-4, 0, 4, 0),
             (-4, 0, 4, 0), (-5, 0, 5, 0), (-5, 0, 5, 0), (-5, 0, 5, 0), 
             (-5, 0, 5, 0)]
twlist0042 = [0.0, -0.5, 0.0, 0.5, 1.0, -0.5, 0.0, 0.5, 1.0]
gflist0042 = [704, 864, 1024, 1184, 1344, 1504, 1664, 1824]

plist001 = plist0042
twlist001 = twlist0042
gflist001 = [500, 564, 628, 692, 756, 820, 884, 948]
gflist001_a = [500, 516, 532, 564, 580, 596, 628, 644,
               660, 692, 708, 724, 756, 772, 788, 820,
               836, 852, 884, 916, 948, 980, 1012]
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
    Data.clear()
    return Data
    
def plot_VmA(data, save=False):
    p.figure()
    #p.title('DSDR')
    p.xlabel('$(ap)^{2}$')
    p.ylabel('$2(\Lambda_V - \Lambda_A)/(\Lambda_V + \Lambda_A)$')
    x = np.array([d.apSq for d in data])
    y = [d.Lambda_VmA for d in data]
    s = [d.Lambda_VmA_sigmaJK for d in data]
    p.xlim(0.6,2.4)
    p.ylim(-.002, .01)
    #p.ylim(10**(-4),.011)
    #p.loglog(x, 0.0034*x**(-3))  # Log-log plot.
    p.errorbar(x, y, s, fmt='o')
    if save:
        root = '/Users/atlytle/Desktop/'
        p.savefig(root + 'VmA_DSDR_chiral_draft.pdf')
    else:
        p.show()
        
def plot_PmS(data, save=False):
    p.figure()
    #p.title('DSDR')
    p.xlabel('$(ap)^{2}$')
    p.ylabel('$2(\Lambda_P - \Lambda_S)/(\Lambda_S + \Lambda_P)$')
    x = np.array([d.apSq for d in data])
    y = [d.Lambda_PmS for d in data]
    s = [d.Lambda_PmS_sigmaJK for d in data]
    p.xlim(0.6,2.4)
    p.ylim(.004,.3)
    p.loglog(x, 0.062*x**(-3))  # Log-log plot. .085
    p.errorbar(x, y, s, fmt='o')
    if save:
        root = '/Users/atlytle/Desktop/'
        p.savefig(root + 'PmS_DSDR_loglog_chiral_draft.pdf')
    else:
        p.show()
        
def to_gnuplot(data):
    'Create columnar data for plots to be consumed by gnuplot.'
    root = '/Users/atlytle/Desktop/DSDR_bilins/'
    dat = np.array([[d.apSq for d in data], 
                   [d.Lambda_PmS for d in data],
                   [d.Lambda_PmS_sigmaJK for d in data],
                   [d.Lambda_VmA for d in data],
                   [d.Lambda_VmA_sigmaJK for d in data]])
    #with open(root+'DSDR_bilins.dat', 'w') as f:
    #    f.write(np.transpose(dat))
    np.savetxt(root+'DSDR_bilins.dat', np.transpose(dat))
    
    
def main():
    
    print "Loading data...",
    data0042 = load_DSDR_Data(.0042, plist0042, twlist0042, gflist0042)
    data001 = \
        load_DSDR_Data(.001, plist001[0:1], twlist001[0:1], gflist001_b) +\
        load_DSDR_Data(.001, plist001[1:2], twlist001[1:2], gflist001) +\
        load_DSDR_Data(.001, plist001[2:3], twlist001[2:3], gflist001_a) +\
        load_DSDR_Data(.001, plist001[3:], twlist001[3:], gflist001)
    print "complete."

    # Compute results.
    print "Computing Zs...",
    pool = Pool()
    data001 = pool.map_async(calc_Zs, data001).get()
    data0042 = pool.map_async(calc_Zs, data0042).get()
    pool.close()
    pool.join()
    print "chiral limit...",
    data0 = map(fits.line_fit_bilinear_Lambdas, data001, data0042)
    print "complete."
    
    print [d.apSq for d in data001]
    print ''
    print [d.Lambda_V for d in data001]
    print [d.Lambda_A for d in data001]
    print ''
    print [d.Lambda[0] for d in data001]
    print [d.Lambda[15] for d in data001]
    
    #plot_PmS(data0, save=True)
    #plot_VmA(data0, save=True)
    to_gnuplot(data0)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
