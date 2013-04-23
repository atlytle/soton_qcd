import pickle, sys
import numpy as np
import IW_analysis
from matching import alpha_s2, alpha_s3, C_178

pickle_root = '/Users/atlytle/Dropbox/pycode/soton/pickle'
DSDR_chiral = pickle_root + '/DSDR_chiral_pickle'
IWf_chiral_15 = pickle_root + '/IWf_chiral_pickle_15'
IWc_chiral_15 = pickle_root + '/IWc_chiral_pickle_15'
IWf_chiral_11 = pickle_root + '/IWf_chiral_pickle_11'
IWc_chiral_11 = pickle_root + '/IWc_chiral_pickle_11'

ZA = 0.68816
dZA = 0.00070

np.set_printoptions(precision=5)

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

def propagate_scalar(A, dA, Z, dZ):
    "Propagate scalar*matrix (scalar) uncertainty."
    return (A*Z, np.sqrt((A*dZ)**2 + (dA*Z)**2))

def new(Zs, full=False):
    "Delta S = 2 --> Delta S = 1."
    convert = np.array([[1, 0, 0],
                        [0, 1, -0.5],
                        [0, -2, 1]])
    if full:
        convert = np.array([[1, 1, -0.5],
                            [1, 1, -0.5],
                            [-2, -2, 1]])
    return convert*Zs[:3,:3]
 
def new2(Zs, full=False):
    "Scale error bars, Delta S = 2 --> Delta S = 1."
    convert = np.array([[1, 0, 0],
                        [0, 1, 0.5],
                        [0, 2, 1]])
    if full:
        convert = np.array([[1, 1, 0.5],
                            [1, 1, 0.5],
                            [2, 2, 1]])
    return convert*Zs[:3,:3]

def print_results(C, ss, dss, Z, dZ, ZA, dZA):
    print 'Z_A: {0} +/- {1}\n'.format(ZA, dZA)
    print 'Z_DSDR/(Z_A)^2:\n{0}\n+/-\n{1}\n'.format(Z, dZ)
    ZA2, dZA2 = propagate_scalar(ZA, dZA, ZA, dZA)
    Z, dZ = propagate_scalar(ZA2, dZA2, Z, dZ)
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
#    with open(IWf_chiral_15, 'r') as f:
#        IWf_chiral_15 = pickle.load(f)
#    with open(IWc_chiral_15, 'r') as f:
#        IWc_chiral_15 = pickle.load(f)
    with open(IWf_chiral_11, 'r') as f:
        IWf_chiral_11 = pickle.load(f)
    with open(IWc_chiral_11, 'r') as f:
        IWc_chiral_11 = pickle.load(f)
    
    ####

    print 'Matching points 1.15 GeV, 3 GeV\n'
    Z_DSDR =  IW_analysis.interpolate_Zs(DSDR_chiral[0], DSDR_chiral[1], 1.1499) 
    # need interpolations here? No that happens in continuum matrix.
    IWc_chiral = IWc_chiral_11
    IWf_chiral = IWf_chiral_11
    
    for scheme in 'gg', 'qq':
        print '____({0}, {1}) - scheme____\n'.format(*scheme)
        Z = Z_DSDR.fourquark_Zs[scheme][:3,:3]
        dZ = Z_DSDR.fourquark_sigmaJK[scheme][:3,:3]
        ss, dss = IW_analysis.continuum_matrix(IWc_chiral,
                                               IWf_chiral, scheme, -19) 
        # ss, and dss develop 'nans' from fits.line_fit(), because
        # the uncertainties on the chirally forbidden elements
        # have been set to zero, which causes a singularity.
        # We fix this by hand, setting the elements back to zero.
        ss[np.isnan(ss)]=0.
        dss[np.isnan(dss)]=0.

        C = C_178(alpha_s3, scheme)
        print_results(C, new(ss), new2(dss), new(Z), new2(dZ), ZA, dZA)

    return 0

if __name__ == "__main__":
    sys.exit(main())
