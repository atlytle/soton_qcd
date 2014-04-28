import sys
import pickle
import numpy as np
from numpy.linalg import inv

import fits, pole_fits
from measurements import chiral_mask
import measurements_exceptional as m

np.set_printoptions(precision=2, suppress=True)

MOM_to_MSbar = np.array([
[1.01716, 0, 0, 0, 0],
[0, 0.977953, -0.13228, 0, 0],
[0, 0.005993, 1.21233, 0, 0],
[0, 0, 0, 1.11023, 0.016719],
[0, 0, 0, 0.0631787, 1.05252]])

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
    
    print "Performing pole subtractions...",
    map(pole_fits.pole_subtract_Data2, data004, data006, data008)
    print "complete."
    
    data_0 = map(fits.line_fit_Data_e, data004, data006, data008)
    
    print np.array([d.mu for d in data_03])
    print np.array([d.mu for d in data004])
        
    print inv(data_005[2].fourquark_Zs)
    print data_005[2].Lambda_VpA
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
