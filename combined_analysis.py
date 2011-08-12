import pickle, sys
import numpy as np
from matching import alpha_s2, alpha_s3, C_178
from IW_analysis import continuum_matrix

pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle'
DSDR_chiral = pickle_root + '/DSDR_chiral_pickle'
IWf_chiral_15 = pickle_root + '/IWf_chiral_pickle_15'
IWc_chiral_15 = pickle_root + '/IWc_chiral_pickle_15'
IWf_chiral_11 = pickle_root + '/IWf_chiral_pickle_11'
IWc_chiral_11 = pickle_root + '/IWc_chiral_pickle_11'


np.set_printoptions(precision=4)

def propagate_errors(Zs, dZs):
    "Uncertainty in matrix multiplication."
    assert len(Zs) == len(dZs) != 0
    Zs2 = [Z*Z for Z in Zs]
    dZs2 = [dZ*dZ for dZ in dZs]
    sigma_sq = np.zeros(Zs[0].shape)

    for i in range(len(Zs)):
        tmp = Zs2[0:i] + dZs2[i:i+1] + Zs2[i+1:]
        sigma_sq += reduce(np.dot, tmp)

    return (reduce(np.dot, Zs), np.sqrt(sigma_sq))

def new(Zs):
    # Delta S = 2 --> Delta S = 1.
    convert = np.array([[1, 0, 0],
                        [0, 1, -0.5],
                        [0, -2, 1]])
    return convert*Zs[:3,:3]
 
def new2(Zs):
    # Scale error bars.
    convert = np.array([[1, 0, 0],
                        [0, 1, 0.5],
                        [0, 2, 1]])
    return convert*Zs[:3,:3]

def print_results(C, ss, dss, Z, dZ):
    print 'Z_DSDR:\n{0}\n+/-\n{1}\n'.format(Z, dZ)
    print 'sigma:\n{0}\n+/-\n{1}\n'.format(ss, dss)
    print 'C:\n{0}\n'.format(C)
    factors = [C, ss, Z]
    sigmas = [np.zeros(3), dss, dZ]
    r, dr = propagate_errors(factors, sigmas)
    print 'Final Result:\n{0}\n+/-\n{1}\n'.format(r, dr)

def main():
    global DSDR_chiral
    global IWf_chiral_15, IWc_chiral_15, IWf_chiral_11, IWc_chiral_11
    # Load data.
    with open(DSDR_chiral, 'r') as f:
        DSDR_chiral = pickle.load(f)
    with open(IWf_chiral_15, 'r') as f:
        IWf_chiral_15 = pickle.load(f)
    with open(IWc_chiral_15, 'r') as f:
        IWc_chiral_15 = pickle.load(f)
    with open(IWf_chiral_11, 'r') as f:
        IWf_chiral_11 = pickle.load(f)
    with open(IWc_chiral_11, 'r') as f:
        IWc_chiral_11 = pickle.load(f)
    
    print 'Matching points 1.5 GeV, 3 GeV\n'
    Z_DSDR =  DSDR_chiral[2]
    IWc_chiral = IWc_chiral_15
    IWf_chiral = IWf_chiral_15

    print '____(g, g) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'gg', -2)
    C = C_178(alpha_s3, 'gg')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)

    print '____(g, q) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs_q[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK_q[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'gq', -2)
    C = C_178(alpha_s3, 'gq')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)

    print '____(q, g) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs_qg[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK_qg[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'qg', -2)
    C = C_178(alpha_s3, 'qg')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)

    print '____(q, q) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs_qq[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK_qq[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'qq', -2)
    C = C_178(alpha_s3, 'qq')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)
    
    ####

    print 'Matching points 1.1 GeV, 3 GeV\n'
    Z_DSDR =  DSDR_chiral[0] 
    IWc_chiral = IWc_chiral_11
    IWf_chiral = IWf_chiral_11

    print '____(g, g) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'gg', -2)
    C = C_178(alpha_s3, 'gg')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)

    print '____(g, q) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs_q[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK_q[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'gq', -2)
    C = C_178(alpha_s3, 'gq')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)

    print '____(q, g) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs_qg[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK_qg[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'qg', -2)
    C = C_178(alpha_s3, 'qg')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)

    print '____(q, q) - scheme____\n'
    Z = new(Z_DSDR.fourquark_Zs_qq[:3,:3])
    dZ = new2(Z_DSDR.fourquark_sigmaJK_qq[:3,:3])
    ss = continuum_matrix(IWc_chiral, IWf_chiral, 'qq', -2)
    C = C_178(alpha_s3, 'qq')
    print_results(C, new(ss[0]), new2(ss[1]), Z, dZ)


    return 0

if __name__ == "__main__":
    sys.exit(main())
