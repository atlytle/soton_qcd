import sys
import pickle
import itertools
import optparse
import numpy as np
import pylab as p
from scipy.interpolate import interp1d
from multiprocessing import Pool

import fits
import combined_analysis
import pyNPR as npr
import measurements as m
import domain_wall as dw
import output as out

# Parameters.
# 32^3 x 64
plistIWf_a = [(-3,0,3,0), (-4,0,4,0), (-4,0,4,0), (-5,0,5,0), (-5,0,5,0)]
twlistIWf_a = [0.25, -0.75, .375, -0.625, 0.375]
gflist004_a = [1700, 1740, 1780, 1820, 1860, 1900, 1940, 1980,
               2020, 2060, 2100, 2140, 2180, 2220, 2260]
gflist006_a = [1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280,
               1320, 1360, 1400, 1440, 1480]
gflist008_a = [1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080, 
               2120, 2160]

plistIWf_b = [(-2,0,2,0)]
twlistIWf_b = [-.413]
gflist004_b = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
              1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650,
              1700, 1750, 1800, 1850, 1900, 1950]
gflist006_b = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650,
               1700, 1750]
gflist008_b = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
              1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]

plistIWf_c = [(-2,0,2,0)]
twlistIWf_c = [.783]
gflist004_c = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]
gflist006_c = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]
gflist008_c = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]

plistIWf_d = [(-5,0,5,0)]
twlistIWf_d = [-0.531292]
gflist004_d = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
gflist006_d = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350]
gflist008_d = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300]

# 24^3 x 64
plistIWc_a = [(-3,0,3,0)]*14 + [(-4,0,4,0)]
twlistIWc_a = [-0.375, -0.1875, 0.1875, 0.375, 0.5625, 0.75, 0.9375,
               1.125, 1.3125, 1.5, 1.6875, 1.875, 2.0625, 2.25, 1.5]
gflist005_a = [1000, 1160, 1320, 1480, 1640, 1800, 1960, 2120,
               2280, 2440, 2600, 2760, 2920, 3080, 3240, 3400,
               3560, 3720, 3880, 4040, 4200, 4360, 4520, 4680,
               4840, 5000, 5160, 5320, 5480, 5640, 5800, 5960,
               6120, 6280, 6440, 6600]
gflist01_a = [1460, 1620, 1780, 1940, 2100, 2260, 2420, 2580,
              2740, 2900, 3060, 3220, 3380, 3540, 3700, 3860,
              4020, 4180, 4340, 4500, 4660, 4820, 4980]
gflist02_a = [1250, 1410, 1570, 1730, 1890, 2050, 2210, 2370,
              2530, 2690, 2850, 3010, 3170, 3330, 3490]
gflist03_a = [1140, 1300, 1460, 1620, 1780, 1940, 2100, 2260,
              2420, 2580, 2740, 2900, 3060, 980]

plistIWc_b = [(-2,0,2,0), (-2,0,2,0)]
twlistIWc_b = [-0.45136, 0.732]
gflist005_b = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
gflist01_b = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200] #, 2300]?
gflist02_b = [700, 750, 800, 850, 900, 950, 1000, 1050]

plistIWc_c = [(-5,0,5,0)]
twlistIWc_c = [-0.632547]
gflist005_c = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
gflist01_c = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
gflist02_c = [700, 750, 800, 850, 900, 950, 1000, 1050]

def load_IWf_Data(m, plist, twlist, gflist):
    return [npr.IWf_Data(m, plist[i], twlist[i], gflist)
            for i in range(len(plist))]

def load_IWc_Data(m, plist, twlist, gflist):
    return [npr.IWc_Data(m, plist[i], twlist[i], gflist)
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

def plot_data(data_, scheme, O, P, legend_spec, save=False, name_spec=''):
    '''Plot Zs vs (ap)^2 at finite am and in chiral limit.'''
    
    mark = ['o', 'o', 'o', 'o-k']
    id=0
    title = {'gg': '\gamma^{\mu}, \gamma^{\mu}', 'gq': '\gamma^{\mu}, q',
             'qg': 'q, \gamma^{\mu}', 'qq': 'q, q'}
    label = str(O+1) + str(P+1)
    legend = ()

    p.figure()
    p.title('$({0}) - \mathrm{{scheme}}$'.format(title[scheme]))
    p.xlabel('$(ap)^{2}$')
    p.ylabel('$ Z_{0} / Z_A^2$'.format('{'+label+'}'), fontsize=16)
    
    for data in data_:
        x, y, s = extract_data(data, scheme, O, P)
        dada = p.errorbar(x, y, yerr=s, fmt=mark[id])
        legend += dada[0],
        id += 1
    p.legend(legend, legend_spec, 'best')
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton'\
               '/SUSY_BK/plots'
        p.savefig(root + '/Z_{0}_{1}_{2}.pdf'.format(name_spec, scheme, label))
    else:
        p.show()


def print_results(data):
    '''Print results for step-scaling data.'''
    for scheme in 'gg', 'qq':
        print "____{0}-scheme____".format(scheme)
        for d in data:
            print "am={0}, mu={1}".format(d.m, d.mu)
            try:
                print "Lambda:"
                print d.Zinv_q
                print "Lambda^{-1}:"
                print d.Z_tmpq
                #print d.step_scale[scheme]
            except:
                print "output error"
            try:
                print "Z:"
                print combined_analysis.new(d.fourquark_Zs[scheme])
                print "ssf:"
                print combined_analysis.new(d.step_scale[scheme])
            except:
                print "output error"
            print ''
            
    
def parse_args():
    parser = optparse.OptionParser()
    parser.add_option('-c', '--compute', action='store_true', dest='compute',
                      help = 'Compute Zs from scratch.')
    parser.add_option('-d', '--dump', action='store_true', dest='dump',
                      help = 'Pickle results of the computation.')
    parser.add_option('-l', '--load', action='store_true', dest='load',
                      help = 'Load Zs from pickle directory.')
    parser.add_option('-p', '--plot', action='store_true', dest='plot',
                      help = 'Plot results.')
    parser.add_option('-s', '--save', action='store_true', dest='save',
                      help = 'Save the plots.')
    options, args = parser.parse_args()
    return options

def main():
    options = parse_args()
   
    global data004, data006, data008, data005, data01, data02
    if options.compute:
        print "Initializing data structures...",
        # Fine IW.
        data004 = load_IWf_Data(.004, plistIWf_a, twlistIWf_a, gflist004_a) +\
                  load_IWf_Data(.004, plistIWf_b, twlistIWf_b, gflist004_b) +\
                  load_IWf_Data(.004, plistIWf_c, twlistIWf_c, gflist004_c) +\
                  load_IWf_Data(.004, plistIWf_d, twlistIWf_d, gflist004_d)

        data006 = load_IWf_Data(.006, plistIWf_a, twlistIWf_a, gflist006_a) +\
                  load_IWf_Data(.006, plistIWf_b, twlistIWf_b, gflist006_b) +\
                  load_IWf_Data(.006, plistIWf_c, twlistIWf_c, gflist006_c) +\
                  load_IWf_Data(.006, plistIWf_d, twlistIWf_d, gflist006_d)

        data008 = load_IWf_Data(.008, plistIWf_a, twlistIWf_a, gflist008_a) +\
                  load_IWf_Data(.008, plistIWf_b, twlistIWf_b, gflist008_b) +\
                  load_IWf_Data(.008, plistIWf_c, twlistIWf_c, gflist008_c) +\
                  load_IWf_Data(.008, plistIWf_d, twlistIWf_d, gflist008_d)

        
        data004.sort(key=lambda D: D.mu)  # Order by energy.
        data006.sort(key=lambda D: D.mu)
        data008.sort(key=lambda D: D.mu)
        
        print [d.mu for d in data008]

        # Coarse IW.
        data005 = load_IWc_Data(.005, plistIWc_a, twlistIWc_a, gflist005_a) +\
                  load_IWc_Data(.005, plistIWc_b, twlistIWc_b, gflist005_b) +\
                  load_IWc_Data(.005, plistIWc_c, twlistIWc_c, gflist005_c)

        data01 = load_IWc_Data(.01, plistIWc_a, twlistIWc_a, gflist01_a) +\
                 load_IWc_Data(.01, plistIWc_b, twlistIWc_b, gflist01_b) +\
                 load_IWc_Data(.01, plistIWc_c, twlistIWc_c, gflist01_c)

        data02 = load_IWc_Data(.02, plistIWc_a, twlistIWc_a, gflist02_a) +\
                 load_IWc_Data(.02, plistIWc_b, twlistIWc_b, gflist02_b) +\
                 load_IWc_Data(.02, plistIWc_c, twlistIWc_c, gflist02_c)

        
        data005.sort(key=lambda D: D.mu)
        data02.sort(key=lambda D: D.mu)
        data01.sort(key=lambda D: D.mu)
        
        print [d.mu for d in data02]

        print "complete"
        
        print "Computing Zs...",
        # Compute results.
        pool = Pool()
        
        # Fine Zs.
        data004 = pool.map_async(calc_Zs, data004).get()
        data006 = pool.map_async(calc_Zs, data006).get()
        data008 = pool.map_async(calc_Zs, data008).get()

        # Coarse Zs.
        data005 = pool.map_async(calc_Zs, data005).get()
        data01 = pool.map_async(calc_Zs, data01).get()
        data02 = pool.map_async(calc_Zs, data02).get()
        #data03 = pool.map_async(calc_Zs, data03).get()

        pool.close()
        pool.join()
        print "complete"

        # Chiral Limits.
        data0c = map(fits.line_fit_Data, data005, data01, data02)
        data0f = map(fits.line_fit_Data, data004, data006, data008)
   
    if options.compute and options.dump:
        print "Pickling data...",    
        pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/SUSY_BK'
        pickle_dict = \
            {'/IWf_chiral_pickle': data0f, '/IWc_chiral_pickle': data0c,
             '/IWf_004_pickle': data004, '/IWf_006_pickle': data006,
             '/IWf_008_pickle': data008, '/IWc_005_pickle': data005,
             '/IWc_01_pickle': data01, '/IWc_02_pickle': data02}

        for name, data in pickle_dict.iteritems():
            with open(pickle_root+name, 'w') as f:
                pickle.dump(data, f)
        
        print "complete"

    if options.load:
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
        
        for d in data004[-4:]:
            print '(ap)^2:', d.apSq
            print 'mu^2:', d.mu*d.mu
            print 'Lambda_A:', d.Lambda_A
            print '  Lambda_V:', d.Lambda_V
            print ''
            
            print 'Zs:'
            print d.fourquark_Zs['gg'], '+/-\n', d.fourquark_sigmaJK['gg']
            print '\n'

    # Plots.
    if options.plot:
        print "Plotting results...",
        coarse_legend = ('$am=.02$', '$am=.01$', '$am=.005$', '$am=-m_{res}$')
        fine_legend = ('$am=.008$', '$am=.006$', '$am=.004$', '$am=-m_{res}$')
        # Plot coarse data.
        for O, P in itertools.product(range(5), range(5)):
            plot_data([data02, data01, data005, data0c], 
                             'qq', O, P, coarse_legend, save, 'coarse')
        # Plot fine data.
        for O, P in itertools.product(range(5), range(5)):
            plot_data([data008, data006, data004, data0f], 
                             'qq', O, P, fine_legend, save, 'fine')
        print "complete"

    print "IW_SUSY_analysis complete."
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
