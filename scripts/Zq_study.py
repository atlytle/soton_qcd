import sys
import pickle
import numpy as np
import pylab as p
from multiprocessing import Pool

import pyNPR as npr
import measurements as m

ar = np.array
np.set_printoptions(precision=2, suppress=True)

# Parameters
# Note these are taken from IW_exceptional_analysis_24/32cube.py
# There may be more data points available for non-exceptional cf IW_analysis.py
# I just study one mass point since m dependence should be very weak.

# 24^3 x 64

plist_IWc = [(-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), 
             (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-4,0,4,0)]
twlist_IWc = [0.1875, 0.375, 0.5625, 0.75, 1.125, 1.5, 1.6875, 2.25, 1.5]
gflist_01 = [1460, 1620, 1780, 1940, 2100, 2260, 2420, 2580, 2740, 2900,
             3060, 3220, 3380, 3540, 3700, 3860, 4020, 4180, 4340, 4500,
             4660, 4820, 4980]
             
# 32^3 x 64
plistIWf = [(-3,0,3,0), (-4,0,4,0), (-4,0,4,0),
            (-5,0,5,0), (-5,0,5,0)]
twlistIWf = [0.25, -0.75, 0.375, -0.625, 0.375]
gflist006 = [1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280,
             1320, 1360, 1400, 1440, 1480]
             
def load_IWf_Data(m, plist, twlist, gflist):
    return [npr.IWf_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]

def load_IWc_Data(m, plist, twlist, gflist):
    return [npr.IWc_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]
            
def calc_Zs(Data):
    Data.load()
    m.bilinear_Lambdas(Data)
    m.bilinear_LambdaJK(Data)
    Data.clear()
    return Data
    
def extract_data(data):
    x = [d.mu for d in data]
    y1 = [d.prop_trace_in for d in data]
    y2 = [d.prop_trace_in2 for d in data]
    return x, y1, y2
    
def plot_data(data_, legend_spec, save=False):
    legend = ()
    p.figure()
    p.xlabel('$\mu$ [GeV]')
    p.ylabel('$Z_q$')
    
    fspec = ['bo', 'bs', 'ko', 'ks']
    i = 0
    
    for data in data_:
        x, y1, y2 = extract_data(data)
        dada = p.errorbar(x, y1, fmt =fspec[i])
        i +=1
        dada = p.errorbar(x, y2, fmt=fspec[i])
        i += 1
        legend += dada[0],
    p.legend(legend, legend_spec, 'best')
        
    if save:
        p.savefig('/Users/atlytle/Desktop/Zq_study.pdf')
    else:
        p.show()
    
def main():
    print 'Loading data...'
    data_01 = load_IWc_Data(.01, plist_IWc, twlist_IWc, gflist_01)
    data_006 = load_IWf_Data(.006, plistIWf, twlistIWf, gflist006)
    
    print 'Computing stuff..'
    pool = Pool()
    
    data_01 = pool.map_async(calc_Zs, data_01).get()   
    data_006 = pool.map_async(calc_Zs, data_006).get()
    
    pool.close()
    pool.join()
    
    print [(d.mu,d.prop_trace_in) for d in data_01]
    
    print [(d.mu,d.prop_trace_in) for d in data_006]
    
    plot_data([data_01, data_006], ['coarse', 'fine'], save=True)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
