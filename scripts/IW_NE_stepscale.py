"""
18-Aug-2014
Non-exceptional Iwasaki step-scale functions in (g,g) and (q,q) schemes,
for the complete Delta S = 2 basis.

These are being used for the SUSY KKbar project.  The pickled data was computed
in IW_SUSY_analysis.py. Some earlier results are in IW_step_scaling.pdf
in my individual postings, and it looks like that data was computed with
IW_analysis.py.
"""

import sys
sys.path.append('../')
import pickle
import itertools
import numpy as np
import pylab as p
from scipy.interpolate import interp1d

import pyNPR as npr
from step_scaling import sigma  # Step-scaling function
from measurements import chiral_mask

np.set_printoptions(precision=4, suppress=True)

def interpolation(data, O, P):
    "Interpolation function for step-scaling data."
    x = [d.mu for d in data]
    y = [d.ss[O][P] for d in data]
    #s = [d.ss_sig[O][P] for d in data]
    x_ = np.linspace(data[0].mu, data[-1].mu)
    y_ = interp1d(x, y, kind='cubic')         # (x_)
    #s_ = interp1d(x, s, kind='cubic')
    
    #return (x_, y_, s_, x, y, s)
    return (x_, y_, x, y)

def plot_data(data_, O, P, save=False):
    "cf IW_E_stepscale.py."
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
        #x_, y_, s_, x, y, s = interpolation(data, O, P)
        x_, y_, x, y = interpolation(data, O, P)
        dada = p.plot(x, y, 'o')#, x_, y_(x_), '-')#,
                      #x_, y_(x_)+s_(x_), '--', x_, y_(x_)-s_(x_), '--')
        legend += dada[0], #dada[1],
        p.setp(dada, color=colors[i])
        i += 1
    
    # # Continuum extrapolate interpolated data.
    # x_, y, s = continuum_extrap(data_[0], data_[1], O, P)
    # dada = p.plot(x_, y, '-')#, x_, y+s, '--', x_, y-s, '--')
    # legend += dada[0],
    # p.setp(dada,  color='green')
        
    p.legend(legend, ('coarse', 'fine'))
    # p.legend(legend, ('$a=0.116 \, \mathrm{fm}$',
    #                   '$a=0.088 \, \mathrm{fm}$', '$a=0$'), 'best') 
        
    # Output.
    if save:
        root = '/Users/atlytle/Dropbox/TeX_docs/soton/SUSY_BK/plots'
        p.savefig(root + '/sigma_{0}.pdf'.format(label))
    else:
        p.show()

def main():
    print "Loading data..",
    pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle/SUSY_BK'
    with open(pickle_root+'/IWc_chiral_pickle', 'r') as f:
        d_0c = pickle.load(f)
    with open(pickle_root+'/IWf_chiral_pickle', 'r') as f:
        d_0f = pickle.load(f)
    print "done."

    print [d.mu for d in d_0c]
    #print [d.Z_chi for d in d_0c]
    print [d.mu for d in d_0f]
    #print [d.Z_chi for d in d_0f]
    # Denominators
    r0c = d_0c[-2]
    r0f = d_0f[-3]
    print r0c.mu
    print r0f.mu

    for d in d_0c:
        d.ss = sigma(d.Z_chi, r0c.Z_chi)
        d.ss_q = sigma(d.fourquark_Zs['qq'], r0c.fourquark_Zs['qq'])

    for d in d_0f:
        d.ss = sigma(d.Z_chi, r0f.Z_chi)
        d.ss_q = sigma(d.fourquark_Zs['qq'], r0f.fourquark_Zs['qq'])

    plot_data([d_0c, d_0f], 0, 0, save=True)
    for O, P in itertools.product([1,2],[1,2]):
        plot_data([d_0c, d_0f], O, P, save=True)
    for O, P in itertools.product([3,4],[3,4]):
        plot_data([d_0c, d_0f], O, P, save=True)


    # print [d.ss for d in d_0c], '\n'
    # print [d.ss_q for d in d_0c], '\n'
    # print [d.ss for d in d_0f], '\n'
    # print [d.ss_q for d in d_0f], '\n'



if __name__ == "__main__":
    sys.exit(main())