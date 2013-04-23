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
from pole_fits import pole_subtract,single_pole_fit

ar = np.array
np.set_printoptions(precision=5, suppress=True)

# Parameters
# 32^3 x 64
plistIWf = [(-3,0,3,0), (-4,0,4,0), (-4,0,4,0),
            (-5,0,5,0), (-5,0,5,0)]
twlistIWf = [0.25, -0.75, 0.375, -0.625, 0.375]
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
    m.fourquark_ZsJK(Data)
    Data.clear()
    return Data
    
def extract_data(data, O, P):
    '''Extract Zs for plotting.'''
    x = [d.apSq for d in data]
    y = [d.fourquark_Zs[O][P] for d in data]
    #s = [d.fourquark_sigmaJK[O][P] for d in data]
    return (x, y)
    
def plot_data(data_, O, P, legend_spec, save=False):
    label = str(O+1) + str(P+1)
    legend = ()
    
    p.figure()
    p.title('$32^3$ IW - exceptional')
    p.xlabel('$(ap)^{2}$')
    p.ylabel('$Z_{0}$'.format('{'+label+'}'), fontsize=16)
    
    for data in data_:
        x, y  = extract_data(data, O, P)
        dada = p.errorbar(x, y, fmt='o')
        legend += dada[0],
    p.legend(legend, legend_spec, 'best')
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton'\
               '/SUSY_BK/exceptional/figs/'
        p.savefig(root + 'Z_{0}_exceptional_32cube.pdf'.format(label))
    else:
        p.show()
        
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
    
def print_out(d):
        print '\n'
        print d.p, d.tw
        print 'ap^2:', d.apSq, 'mu:', d.mu
        print d.Zsub
        print ''    
        print d.Zsub_sigmaJK

def main():
    options = parse_args()
    if options.compute:
        print "Computing Zs..."
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
        
        print "complete."
        
    root = '/Users/atlytle/Dropbox/pycode/soton/pickle/IW_exceptional'
    
    if options.compute and options.dump:
        print "Pickling data..."
        with open(root+'/IWf_exceptional_004.pkl', 'w') as f:
            pickle.dump(data004, f)
        with open(root+'/IWf_exceptional_006.pkl', 'w') as f:
            pickle.dump(data006, f)
        with open(root+'/IWf_exceptional_008.pkl', 'w') as f:
            pickle.dump(data008, f)
            
    if options.load:
        print "Un-pickling data..."
        with open(root+'/IWf_exceptional_004.pkl', 'r') as f:
            data004 = pickle.load(f)
        with open(root+'/IWf_exceptional_006.pkl', 'r') as f:
            data006 = pickle.load(f)
        with open(root+'/IWf_exceptional_008.pkl', 'r') as f:
            data008 = pickle.load(f)
            
        for d in data004[-2:]:
            print '(ap)^2:', d.apSq
            print 'mu^2:', d.mu*d.mu
            print 'Lambda_A:', d.Lambda_A, 
            print '  Lambda_V:', d.Lambda_V
            print ''
            
            print 'Zs:'
            print d.fourquark_Zs, '+/-\n', d.fourquark_sigmaJK
            print '\n\n'

        print "Computing chiral limits...",
        data_0 = map(fits.line_fit_Data_e, data004, data006, data008)

        print "Performing pole subtractions...",
        map(pole_fits.pole_subtract_Data2, data004, data006, data008)
        # Cobble together bits of Zinv and return Z for each m.
        for x in data004, data006, data008:
            map(m.Zsub, x)
        # Linear extrapolate results.
        data_sub = map(fits.line_fit_Data_e, data004, data006, data008)
        
        #print_out(data_sub[-1])
        #print_out(data_sub[-2])
        
    # Plots.
    if options.plot:
        legend_spec = ('$am=0.008$', '$am=0.006$', '$am=0.004$')
        for O, P in itertools.product(range(5), range(5)):
            plot_data([data008, data006, data004], O, P, legend_spec, save)
        return 0
    return 0

if __name__ == "__main__":
    sys.exit(main())
