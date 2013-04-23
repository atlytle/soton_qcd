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

ar = np.array
np.set_printoptions(precision=5, suppress=True)

# Parameters
# 24^3 x 64

plist_IWc = [(-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), 
             (-3,0,3,0), (-3,0,3,0), (-3,0,3,0), (-4,0,4,0)]
twlist_IWc = [0.1875, 0.375, 0.5625, 0.75, 1.125, 1.5, 1.6875, 2.25, 1.5]
gflist_03 = [980, 1140, 1300, 1460, 1620, 1780, 1940, 2100, 2260, 2420,
             2580, 2740, 2900, 3060]
gflist_02 = [1250, 1410, 1570, 1730, 1890, 2050, 2210, 2370, 2530, 2690,
             2850, 3010, 3170, 3330, 3490]
gflist_01 = [1460, 1620, 1780, 1940, 2100, 2260, 2420, 2580, 2740, 2900,
             3060, 3220, 3380, 3540, 3700, 3860, 4020, 4180, 4340, 4500,
             4660, 4820, 4980]
gflist_005 = [1000, 1160, 1320, 1480, 1640, 1800, 1960, 2120,
              2280, 2440, 2600, 2760, 2920, 3080, 3240, 3400, 3560, 3720,
              3880, 4040, 4200, 4360, 4520, 4680, 4840, 5000, 5160, 5320,
              5480, 5640, 5800, 5960, 6120, 6280, 6440, 6600]

def load_IWc_Data(m, plist, twlist, gflist):
    return [npr.IWc_Exceptional_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]
            
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
    s = [d.fourquark_sigmaJK[O][P] for d in data]
    return x, y, s
        
def plot_data(data_, O, P, legend_spec, save=False):
    label = str(O+1) + str(P+1)
    legend = ()
    
    p.figure()
    p.title('$24^3$ IW - exceptional')
    p.xlabel('$(ap)^{2}$')
    p.ylabel('$Z_{0}$'.format('{'+label+'}'), fontsize=16)
    
    for data in data_:
        x, y, s  = extract_data(data, O, P)
        dada = p.errorbar(x, y, fmt='o')
        legend += dada[0],
    p.legend(legend, legend_spec, 'best')
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton'\
               '/SUSY_BK/exceptional/figs/'
        p.savefig(root + 'Zsub_{0}_exceptional_24cube.pdf'.format(label))
    else:
        p.show()
        
def plot_chiral_extrap(data_, pspec, O, P, save=False):
    '''Pull out data at a given momentum (pspec) and plot vs. am.'''
    label = str(O+1) + str(P+1)
    legend = ()
    
    p.figure()
    p.title('$24^3$ IW - exceptional')
    p.xlabel('$am$')
    p.ylabel('$Z_{0}$'.format('{'+label+'}'), fontsize=16)
    data = [d[pspec] for d in data_]
    for d in data:
        x, y, s = d.m, d.fourquark_Zs[O][P], d.fourquark_sigmaJK[O][P]
        p.errorbar(x, y, s, fmt='bo')
    # Assumes chiral data w/ fit parameters is first element.
    d = data[0]
    x = np.linspace(-d.mres, 0.02)
    p.plot(x, d.Zfit.a[O][P] + d.Zfit.b[O][P]*(x+d.mres), 'k--')
    if save:
        pass
    else:
        p.show()
        
def plot_chiral_extrap_Zinv(data_, pspec, O, P, legend_spec, save=False):
    '''Pull out data at a given momentum (pspec) and plot vs. am.'''
    label = str(O+1) + str(P+1)
    legend = ()
    
    p.figure()
    p.title('$24^3$ IW - exceptional')
    p.xlabel('$am$')
    p.ylabel('$(Z^{{-1}})_{0}$'.format('{'+label+'}'), fontsize=16)
    data = [d[pspec] for d in data_]
    for d in data:
        x, y, s = d.m, d.Zinv[O][P], d.Zinv_sigmaJK[O][P]
        p.errorbar(x, y, s, fmt='bo')
        x, y, s = d.m, d.Zinv_sub[O,P], d.Zinv_sub_sigma[O][P]
        p.errorbar(x, y, s, fmt='ko')
    # Assumes chiral data w/ fit parameters is first element.
    d = data[0]
    x = np.linspace(-d.mres, 0.02)
    tmp = p.plot(x, d.Zinvfit.a[O][P] + d.Zinvfit.b[O][P]*(x+d.mres), 'b--')
    legend += tmp[0],
    tmp = p.plot(x, d.Zinv_subfit.a[O][P] + d.Zinv_subfit.b[O][P]*(x+d.mres), 'k-')
    legend += tmp[0],
    p.legend(legend, legend_spec, 'best')
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton'\
               '/SUSY_BK/exceptional/figs/'
        p.savefig(root + 'Zinv_sub_{0}_exceptional_24cube.pdf'.format(label))
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
        print "Computing Zs...",
        
        data_03 = load_IWc_Data(.03, plist_IWc, twlist_IWc, gflist_03)
        data_02 = load_IWc_Data(.02, plist_IWc, twlist_IWc, gflist_02)
        data_01 = load_IWc_Data(.01, plist_IWc, twlist_IWc, gflist_01)
        data_005 = load_IWc_Data(.005, plist_IWc, twlist_IWc, gflist_005)
        
        pool = Pool()
        data_03 = pool.map_async(calc_Zs, data_03).get()
        data_02 = pool.map_async(calc_Zs, data_02).get()
        data_01 = pool.map_async(calc_Zs, data_01).get()   
        data_005 = pool.map_async(calc_Zs, data_005).get()    
         
        pool.close()
        pool.join()
        
        print "complete."
        
        print "Calculating chiral limit...",
        # Naive chiral limits.
        data_0_3pt = map(fits.line_fit_Data_e, data_005, data_01, data_02)
        data_0_4pt = map(fits.line_fit_Data_e, data_005, data_01, 
                                               data_02, data_03)
        print "complete."
   
    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/'\
                  'IW_exceptional_24cube/'      
    if options.compute and options.dump:
        print "Pickling data...",  
                      
        with open(pickle_root+'IW_exceptional_24cube_03.pkl', 'w') as f:
            pickle.dump(data_03, f)   
        with open(pickle_root+'IW_exceptional_24cube_02.pkl', 'w') as f:
            pickle.dump(data_02, f)  
        with open(pickle_root+'IW_exceptional_24cube_01.pkl', 'w') as f:
            pickle.dump(data_01, f)  
        with open(pickle_root+'IW_exceptional_24cube_005.pkl', 'w') as f:
            pickle.dump(data_005, f)            
        print "complete."    

    if options.load:
        print "Un-pickling data..."
        
        with open(pickle_root+'IW_exceptional_24cube_03.pkl', 'r') as f:
            data_03 = pickle.load(f)
        with open(pickle_root+'IW_exceptional_24cube_02.pkl', 'r') as f:
            data_02 = pickle.load(f)
        with open(pickle_root+'IW_exceptional_24cube_01.pkl', 'r') as f:
            data_01 = pickle.load(f)  
        with open(pickle_root+'IW_exceptional_24cube_005.pkl', 'r') as f:
            data_005 = pickle.load(f)                     
        print "complete."
       
        for d in data_005[-3:]:
            print '(ap)^2:', d.apSq
            print 'mu^2:', d.mu*d.mu
            print 'Lambda_A:', d.Lambda_A
            print '  Lambda_V:', d.Lambda_V
            print ''
            
            print 'Zs:'
            print d.fourquark_Zs, '+/-\n', d.fourquark_sigmaJK
            print '\n'
                
        print "Calculating naive chiral limit...",
        # Naive chiral limits.
        data_0_3pt = map(fits.line_fit_Data_e, data_005, data_01, data_02)
        data_0_4pt = map(fits.line_fit_Data_e, data_005, data_01, 
                                               data_02, data_03)
    
        print "Performing pole subtractions...",
#        data_sub1 = map(pole_fits.pole_subtract_Data, data_01, data_02)
#        data_sub2 = map(pole_fits.pole_subtract_Data, data_005, data_01)
#        data_0_sub = map(fits.line_fit_Data_e, data_sub2, data_sub1)
#        print "complete"
        

        map(pole_fits.pole_subtract_Data2, data_005, data_01, data_02)
        # Cobble together bits of Zinv and return Z for each m.
        for x in data_005, data_01, data_02:
            map(m.Zsub, x)
        # Linear extrapolate results.
        data_sub = map(fits.line_fit_Data_e, data_005, data_01, data_02)
        
        
        #print_out(data_sub[-1])
        #print_out(data_sub[-2])
            
        print "complete."  

    if options.plot:
#        legend_spec = ('$am=0.03$', '$am=0.02$', '$am=0.01$', '$am=0.005$',
#                       '$am = -am_{res}$ (3 pt)', '$am = -am_{res}$ (4 pt)')
#        for O, P in itertools.product(range(5), range(5)):
#            plot_data([data_03, data_02, data_01, data_005, 
#                      data_0_3pt, data_0_4pt], O, P, legend_spec, options.save)
        #legend_spec = ('naive_chiral', 'subtracted')
        for O, P in itertools.product(range(5), range(5)):
            #plot_data([data_0_3pt, data_0_sub], O, P, legend_spec, options.save)
            #plot_chiral_extrap([data_0_3pt, data_005, data_01, data_02, data_03],
            #                    4,O,P)
            plot_chiral_extrap_Zinv([data_sub, data_005, data_01, data_02], 
                                    -1, O, P, ('naive','sub'), options.save)
        
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
