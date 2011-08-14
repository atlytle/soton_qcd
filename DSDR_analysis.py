import sys
import pickle
import numpy as np
import pylab as p
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

def extract_data(data, scheme, O, P):
    '''Extract Zs for plotting.'''
    x = [d.apSq for d in data]
    y = [d.fourquark_Zs[scheme][O][P] for d in data]
    s = [d.fourquark_sigmaJK[scheme][O][P] for d in data]
    return (x, y, s)
def plot_data(data_, scheme, O, P, save=False):
    '''Plot Zs vs (ap)^2 at finite am and in chiral limit.'''
    
    mark = ['o', 'o', 'o-']
    id=0
    title = {'gg': '\gamma^{\mu}, \gamma^{\mu}', 'gq': '\gamma^{\mu}, q',
             'qg': 'q, \gamma^{\mu}', 'qq': 'q, q'}
    label = str(O+1) + str(P+1)
    legend = ()

    p.figure()
    p.title('$({0}) - \mathrm{{scheme}}$'.format(title[scheme]))
    p.xlabel('$(ap)^{2}$')
    p.ylabel('$Z_{0}$'.format('{'+label+'}'), fontsize=16)
    
    for data in data_:
        x, y, s = extract_data(data, scheme, O, P)
        dada = p.errorbar(x, y, yerr=s, fmt=mark[id])
        legend += dada[0],
        id += 1
    p.legend(legend, ('$am=.0042$', '$am=.001$', '$am=-m_{res}$'), 'best')
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton'\
               '/AuXDet_NPR/fourFermi/plots'
        p.savefig(root + '/Z_{0}_{1}.pdf'.format(scheme, label))
    else:
        p.show()

def main():
    
    compute = False
    dump = False
    load = True
    plot = True
    save = False

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
    DSDR_0042 = pickle_root + '/DSDR_0042_pickle'
    DSDR_001 = pickle_root + '/DSDR_001_pickle'
    DSDR_chiral = pickle_root + '/DSDR_chiral_pickle'

    if dump:
        with open(DSDR_0042, 'w') as f:
            pickle.dump(data0042, f)
        with open(DSDR_001, 'w') as f:
            pickle.dump(data001, f)
        with open(DSDR_chiral, 'w') as f:
            pickle.dump(data0, f)


    if load:
        with open(DSDR_0042, 'r') as f:
            data0042 = pickle.load(f)
        with open(DSDR_001, 'r') as f:
            data001 = pickle.load(f)
        with open(DSDR_chiral, 'r') as f:
            data0 = pickle.load(f)

    plot_data([data0042, data001, data0], 'gg', 2, 1)
    
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
