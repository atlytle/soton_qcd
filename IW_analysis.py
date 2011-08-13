import sys
import pickle
import numpy as np
import pylab as p
from scipy.interpolate import interp1d
from multiprocessing import Pool

import fits
import pyNPR as npr
import measurements as m
import domain_wall as dw
import output as out
import step_scaling as ss
from step_scaling import do_ss, do_ssJK

# Parameters.
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

plistIWf_ = [(-2, 0, 2, 0), (-2, 0, 2, 0)]
twlistIWf_ = [-.413, .783]
gflist004_ = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
              1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]
gflist006_ = [1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 
              1400, 1450, 1500, 1550, 1600, 1650]
gflist008_ = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
             1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]


'''plistIW3 = [(-3, 0, 3, 0)]
twlistIW3 = [.135]
gflistIW3 = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250]'''

# 24^3 x 64
plistIWc = [(-3,0,3,0)]*14 + [(-4,0,4,0)]
twlistIWc = [-0.375, -0.1875, 0.1875, 0.375, 0.5625, 0.75, 0.9375,
             1.125, 1.3125, 1.5, 1.6875, 1.875, 2.0625, 2.25, 1.5]
gflist005 = [1000, 1160, 1320, 1480, 1640, 1800, 1960, 2120,
             2280, 2440, 2600, 2760, 2920, 3080, 3240, 3400,
             3560, 3720, 3880, 4040, 4200, 4360, 4520, 4680,
             4840, 5000, 5160, 5320, 5480, 5640, 5800, 5960,
             6120, 6280, 6440, 6600]
gflist01 = [1460, 1620, 1780, 1940, 2100, 2260, 2420, 2580,
            2740, 2900, 3060, 3220, 3380, 3540, 3700, 3860,
            4020, 4180, 4340, 4500, 4660, 4820, 4980]
gflist02 = [1250, 1410, 1570, 1730, 1890, 2050, 2210, 2370,
            2530, 2690, 2850, 3010, 3170, 3330, 3490]
gflist03 = [1140, 1300, 1460, 1620, 1780, 1940, 2100, 2260,
            2420, 2580, 2740, 2900, 3060, 980]

plistIWc_ = [(-2,0,2,0), (-2,0,2,0)]
twlistIWc_ = [-0.45136, 0.732]
gflist005_ = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
gflist01_ = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300]
gflist02_ = [700, 750, 800, 850, 900, 950, 1000, 1050]

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

ss_dict = {'gg': 'step_scale', 'gq': 'step_scale_q',
           'qg': 'step_scale_qg', 'qq': 'step_scale_qq'}
ss_sigma_dict = {'gg': 'step_scale_sigma', 'gq': 'step_scale_sigma_q',
                 'qg': 'step_scale_sigma_qg', 'qq': 'step_scale_sigma_qq'}

def interpolation(data, scheme, O, P):
    '''Interpolating function for step-scaling data.'''

    assert scheme in 'gg', 'gq', 'qg', 'qq'

    x = [d.mu for d in data]
    y = [d.step_scale[scheme][O][P] for d in data]
    s = [d.step_scale_sigma[scheme][O][P] for d in data]
    x_ = np.linspace(data[0].mu, data[-1].mu)
    y_ = interp1d(x, y, kind='cubic')  # (x_)
    s_ = interp1d(x, s, kind='linear')  # (x_)
    
    return (x_, y_, s_, x, y, s)

def continuum_extrap(datac, dataf, scheme, O, P):
    '''Continuum extrapolation of step-scaling data.'''
    
    x_, yc_, sc_, xc, yc, sc = interpolation(datac, scheme, O, P)
    yf_, sf_, xf, yf, sf = interpolation(dataf, scheme, O, P)[1:]

    ac = datac[0].a
    af = dataf[0].a

    pts = [[(ac*ac, yc_(mu), sc_(mu)), (af*af, yf_(mu), sf_(mu))]
           for mu in x_]

    results = map(fits.line_fit, pts)
    y = np.array([r[0] for r in results])
    s = np.array([r[1] for r in results])
    #print 'mu=', x_[-2]
    #print y[-2], s[-2]
    
    return (x_, y, s)

def continuum_matrix(datac, dataf, scheme, x):
    '''Continuum step-scaling matrix at position x.'''
    ymu = lambda O, P: continuum_extrap(datac, dataf, scheme, O, P)[1][x]
    smu = lambda O, P: continuum_extrap(datac, dataf, scheme, O, P)[2][x]

    y = np.array([[ymu(0,0), ymu(0,1), ymu(0,2)],
                  [ymu(1,0), ymu(1,1), ymu(1,2)],
                  [ymu(2,0), ymu(2,1), ymu(2,2)]])
    
    s = np.array([[smu(0,0), smu(0,1), smu(0,2)],
                  [smu(1,0), smu(1,1), smu(1,2)],
                  [smu(2,0), smu(2,1), smu(2,2)]])

    return (y, s)

def plot_data(data_, scheme, O, P, save=False):
    '''Plot of step-scale function vs mu.'''

    assert(len(data_) == 2)  # coarse, fine 
    colors = ['r', 'b']
    id=0
    p.figure()
    title = {'gg': '\gamma^{\mu}, \gamma^{\mu}', 'gq': '\gamma^{\mu}, q',
            'qg': 'q, \gamma^{\mu}', 'qq': 'q, q'}
    p.title('$({0}) - \mathrm{{scheme}}$'.format(title[scheme]))
    p.xlabel('$\mu \, (\mathrm{GeV})$', fontsize=16)
    label = str(O+1) + str(P+1)
    p.ylabel('$\sigma_{0}$'.format('{'+label+'}'), fontsize=16)
    legend = ()

    for data in data_:
        x_, y_, s_, x, y, s = interpolation(data, scheme, O, P)
        #dada = p.plot(*interpolation(data, scheme, O, P))
        dada = p.plot(x, y, 'o', x_, y_(x_), '-',
                      x_, y_(x_)+s_(x_), '--', x_, y_(x_)-s_(x_), '--')
        legend += dada[1],
        p.setp(dada, color=colors[id])
        id = id + 1
    
    x_, y, s = continuum_extrap(data_[0], data_[1], scheme, O, P)
    dada = p.plot(x_, y, '-', x_, y+s, '--', x_, y-s, '--')
    legend += dada[0],
    p.setp(dada,  color='green')
    p.legend(legend, ('$a=0.116 \, \mathrm{fm}$',
                      '$a=0.088 \, \mathrm{fm}$', '$a=0$'), 'best') 
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton/step-scaling/plots'
        p.savefig(root + '/sigma_{0}_{1}.pdf'.format(scheme, label))
    else:
        p.show()

def main():
    compute = True
    dump = True
    load = False
    plot = False
    save = False

    if compute:
        # Load data.
        data004 = load_IWf_Data(.004, plistIWf_, twlistIWf_, gflist004_) +\
                  load_IWf_Data(.004, plistIWf, twlistIWf, gflist004)

        data006 = load_IWf_Data(.006, plistIWf_, twlistIWf_, gflist006_) +\
                  load_IWf_Data(.006, plistIWf, twlistIWf, gflist006)

        data008 = load_IWf_Data(.008, plistIWf_, twlistIWf_, gflist008_) +\
                  load_IWf_Data(.008, plistIWf, twlistIWf, gflist008)

        data005 = load_IWc_Data(.005, plistIWc_, twlistIWc_, gflist005_) +\
                  load_IWc_Data(.005, plistIWc, twlistIWc, gflist005)
        data01 = load_IWc_Data(.01, plistIWc_, twlistIWc_, gflist01_) +\
                 load_IWc_Data(.01, plistIWc, twlistIWc, gflist01)
        data02 = load_IWc_Data(.02, plistIWc_, twlistIWc_, gflist02_) +\
                 load_IWc_Data(.02, plistIWc, twlistIWc, gflist02)
        #data03 = load_IWc_Data(.03, plistIWc, twlistIWc, gflist03)

                 
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

        # Chiral Limits.
        data0c = map(fits.line_fit_Data, data005, data01, data02)
        data0f = map(fits.line_fit_Data, data004, data006, data008)
     
        # Step-scaling functions.

        # why not take chiral limit of step-scale functions??
        # would this not have less m dependence? A: need booststrap
        map(do_ss(data0c[0]), data0c)
        map(do_ssJK(data0c[0]), data0c)
        
        map(do_ss(data0f[0]), data0f)
        map(do_ssJK(data0f[0]), data0f)
    
    
    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle'
    kosher_f = pickle_root + '/IWf_chiral_pickle_11'
    kosher_c = pickle_root + '/IWc_chiral_pickle_11'

    if compute and dump:
        with open(kosher_f, 'w') as f:
            pickle.dump(data0f, f)
        with open(kosher_c, 'w') as f:
            pickle.dump(data0c, f)

    if load:
        with open(kosher_f, 'r') as f:
            data0f = pickle.load(f)
        with open(kosher_c, 'r') as f:
            data0c = pickle.load(f)
    
    print continuum_matrix(data0c, data0f, 'gg', -2)

    #plots
    if plot:
        plot_data([data0c, data0f], 'qg', 0, 0, save)
        plot_data([data0c, data0f], 'qg', 1, 1, save)
        plot_data([data0c, data0f], 'qg', 2, 2, save)
        plot_data([data0c, data0f], 'qg', 1, 2, save)
        plot_data([data0c, data0f], 'qg', 2, 1, save)

    return 0

if __name__ == "__main__":
    sys.exit(main())
