import sys
import pickle
import numpy as np
import pylab as p
from scipy.interpolate import interp1d

import fits, pole_fits
from measurements import chiral_mask

# Step-scale based on IW_analysis.py and step_scaling.py.
# Data was calculated using IW_exceptional_analysis_24/32cube.py

def sigma(Zinv_f, Zinv_i):
    "Step scale sigma."
    Z_f = np.linalg.inv(Zinv_f*chiral_mask)
    Zinv_i = Zinv_i*chiral_mask
    return np.dot(Z_f, Zinv_i)
    
def interpolation(data, O, P):
    "Interpolation function for step-scaling data."
    x = [d.mu for d in data]
    y = [d.step_scale[O][P] for d in data]
    s = [d.step_scale_sig[O][P] for d in data]
    x_ = np.linspace(data[0].mu, data[-1].mu)
    y_ = interp1d(x, y, kind='cubic')         # (x_)
    s_ = interp1d(x, s, kind='cubic')
    
    return (x_, y_, s_, x, y, s)
    
def continuum_extrap(datac, dataf, O, P):
    "Continuum extrapolation of step-scaling data."
    yc_, sc_, xc, yc, sc = interpolation(datac, O, P)[1:]
    yf_, sf_, xf, yf, sf = interpolation(dataf, O, P)[1:]
    x_ = np.linspace(max(xc[0],xf[0]), min(xc[-1],xf[-1]), 400)
    #print x_
    ac = datac[0].a
    af = dataf[0].a

    pts = [[(ac*ac, yc_(mu), sc_(mu)), (af*af, yf_(mu), sf_(mu))]
           for mu in x_]

    results = map(fits.line_fit, pts)
    y = np.array([r[0] for r in results])
    s = np.array([r[1] for r in results])
    
    return (x_, y, s)
    
def plot_data(data_, O, P, save=False):
    assert(len(data_)==2)  # Coarse, fine.
    colors = ['r', 'b']
    i = 0
    p.figure()
    p.xlabel('$\mu \, (\mathrm{GeV})$', fontsize=16)
    label = str(O+1) + str(P+1)
    p.ylabel('$\sigma_{0}$'.format('{'+label+'}'), fontsize=16)
    legend = ()
    
    # Interpolate raw data.
    for data in data_:
        x_, y_, s_, x, y, s = interpolation(data, O, P)
        dada = p.plot(x, y, 'o', x_, y_(x_), '-')#,
                      #x_, y_(x_)+s_(x_), '--', x_, y_(x_)-s_(x_), '--')
        legend += dada[1],
        p.setp(dada, color=colors[i])
        i += 1
    
    # Continuum extrapolate interpolated data.
    x_, y, s = continuum_extrap(data_[0], data_[1], O, P)
    dada = p.plot(x_, y, '-')#, x_, y+s, '--', x_, y-s, '--')
    legend += dada[0],
    p.setp(dada,  color='green')
        
    p.legend(legend, ('$a=0.116 \, \mathrm{fm}$',
                      '$a=0.088 \, \mathrm{fm}$', '$a=0$'), 'best') 
        
    # Output.
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton/SUSY_BK/exceptional/figs'
        p.savefig(root + '/sigma_exceptional_{0}.pdf'.format(label))
    else:
        p.show()
    

def main():
    print "Loading 24^3 IW exceptional data...",
    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/'\
                  'IW_exceptional_24cube/'
                  
    with open(pickle_root +'IW_exceptional_24cube_03.pkl', 'r') as f:
        data_03 = pickle.load(f)
    with open(pickle_root +'IW_exceptional_24cube_02.pkl', 'r') as f:
        data_02 = pickle.load(f)
    with open(pickle_root +'IW_exceptional_24cube_01.pkl', 'r') as f:
        data_01 = pickle.load(f)  
    with open(pickle_root +'IW_exceptional_24cube_005.pkl', 'r') as f:
        data_005 = pickle.load(f)  
    print "complete."
        
    print "Loading 32^3 IW exceptional data...",
    root = '/Users/atlytle/Dropbox/pycode/soton/pickle/IW_exceptional'        
    with open(root+'/IWf_exceptional_004.pkl', 'r') as f:
        data004 = pickle.load(f)
    with open(root+'/IWf_exceptional_006.pkl', 'r') as f:
        data006 = pickle.load(f)
    with open(root+'/IWf_exceptional_008.pkl', 'r') as f:
        data008 = pickle.load(f)
    print "complete."
    
    print [d.mu for d in data_005]
    print [d.mu for d in data008]
    
    print "Performing pole subtractions...",
    map(pole_fits.pole_subtract_Data2, data_005, data_01, data_02)
    map(pole_fits.pole_subtract_Data2, data004, data006, data008)
    print "complete."
    
    print "Taking chiral limits...",
    data_0c = map(fits.line_fit_Data_e, data_005, data_01, data_02)
    data_0f = map(fits.line_fit_Data_e, data004, data006, data008)
    print "complete."
    
    denomC = data_0c[0]
    denomF = data_0f[0]
    Zinv_i = denomC.Zinv_sub
    Zinv_i_sig = denomC.Zinv_sub_sigma
    #print Zinv_i
    #print Zinv_i_sig
    for d in data_0c:
        #print d.mu
        #print sigma(d.Zinv_sub, Zinv_i)
        d.step_scale = sigma(d.Zinv_sub, Zinv_i)
        d.step_scale_sig = sigma(d.Zinv_sub, Zinv_i_sig)  # Naive error.
        
    Zinv_i = denomF.Zinv_sub
    Zinv_i_sig = denomF.Zinv_sub_sigma
    for d in data_0f:
        #print d.mu
        #print sigma(d.Zinv_sub, Zinv_i)
        d.step_scale = sigma(d.Zinv_sub, Zinv_i)
        d.step_scale_sig = sigma(d.Zinv_sub, Zinv_i_sig)  # Naive error.
        
    for O in range(5):
        for P in range(5):
            plot_data([data_0c, data_0f], O, P, save=False)
    
    
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
