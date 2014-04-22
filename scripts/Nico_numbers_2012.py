'''
May 29, 2012
Nicolas' numbers for Z^{chi} \Lambda^{f} F^{-1} in the Delta S=2 basis.'''

import numpy as np
from numpy import array
from numpy.linalg import inv

# DSDR results at ~1.4 GeV.
ID_14_qq = array([
[1., -.00160, .00794],
[.00234, 1., 0.],
[.00077, 0., 1.]])

ID_14_qq_sigma = array([
[0., .00186, .00355],
[.00173, 0., 0.],
[.00222, 0., 0.]])

# Coarse IW results at ~1.4 GeV.
IWc_14_qq = array([
[1., -.00146, .01478],
[.00058, 1., 0.],
[.00224, 0., 1.]])

IWc_14_qq_sigma = array([
[0., .00124, .00170],
[.00091, 0., 0.],
[.00091, 0., 0.]])

# Fine IW results at ~1.4 GeV.
IWf_14_qq = array([
[1., -.00004, .01124],
[.00290, 1., 0.],
[.00377, 0., 1.]])

IWf_14_qq_sigma = array([
[0., .00140, .00409],
[.00094, 0., 0.],
[.00127, 0., 0.]])

# Fine IW results at 3 GeV.
IWf_3_qq = array([
[1., -.00002, -.00004],
[0.00000, 1., 0.],
[.00002, 0., 1.]])

IWf_3_qq_sigma = array([
[0., .00002, .00004],
[.00002, .00000, .00000],
[.00001, .00000, .00000]])

# Delta S=2 --> Delta S=1 conversion matrix.
M = array([
[1., 0., 0.],
[0., 1., 0.],
[0., 0., -2.]])

def convert(matrix):
    "Convert from Delta S=2 to Delta S=1 basis."
    return np.dot(np.dot(M, matrix), inv(M))

def main():
    np.set_printoptions(suppress=True, precision=5)  # Decimal form.
    print "DSDR ~1.4 GeV (q,q):\n", convert(ID_14_qq), "\n +/- \n",\
                                    convert(ID_14_qq_sigma), "\n"
                                    
    print "IW coarse ~1.4 GeV (q,q):\n", convert(IWc_14_qq), "\n +/- \n",\
                                         convert(IWc_14_qq_sigma), "\n"
                                         
    print "IW fine ~1.4 GeV (q,q):\n", convert(IWf_14_qq), "\n +/- \n",\
                                       convert(IWf_14_qq_sigma), "\n"
                                       
    print "IW fine ~3.0 GeV (q,q):\n", convert(IWf_3_qq), "\n +/- \n",\
                                       convert(IWf_3_qq_sigma), "\n"
                                       
    print convert(np.ones((3,3)))
    return 0

if __name__ == "__main__":
    main()
