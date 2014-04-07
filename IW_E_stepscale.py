import sys
import pickle
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
    
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
