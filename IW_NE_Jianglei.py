import sys
import pickle
import numpy as np
from numpy.linalg import inv

import fits
import measurements as m

np.set_printoptions(precision=4, suppress=True)

SMOM_to_MSbar = np.array([
[1.00414, 0, 0],
[0, 1.00084, 0.00507822],
[0, 0.015723, 1.08789]])

# Delta S=2 --> Delta S=1 conversion matrix.
M = np.array([
[1., 0., 0., 0., 0.],
[0., 1., 0., 0., 0.],
[0., 0., -2., 0., 0.],
[0., 0., 0., 1., 0.],
[0., 0., 0., 0., 1.]])

def convert(matrix):
    "Convert from Delta S=2 to Delta S=1 basis."
    return np.dot(np.dot(M, matrix), inv(M))

def main():

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
    
    #print np.array([d.mu for d in data008])
    #print np.array([d.mu for d in data02])
    
    print '------------ 24^3 results, (g, q) scheme ----------'
    for x in 6, 7, -2:
        print 'mu:',data0c[x].mu, ' latt mom:', data0c[x].p, data0c[x].tw
        print convert(data0c[x].fourquark_Zs['gq'])[0:3,0:3], '\n'
        
    print '------------ 32^3 results, (g, q) scheme ----------'
    for x in 2, 3, 5:
        print 'mu:', data0f[x].mu, ' latt mom:', data0f[x].p, data0f[x].tw
        print convert(data0f[x].fourquark_Zs['gq'])[0:3,0:3], '\n'
    #print data004[-3].Lambda_VpA
    #print data004[-3].Vq
    
if __name__ == "__main__":
    sys.exit(main())
