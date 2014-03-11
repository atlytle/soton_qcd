import sys
import pickle
import numpy as np
from multiprocessing import Pool

import fits
import pyNPR as npr
import measurements_exceptional as me
import measurements as m
from matching import Cm, C_S, alpha_s2, alpha_s3

ar = np.array
np.set_printoptions(precision=2, suppress=True)

# Parameters
# Note these are taken from IW_exceptional_analysis_24/32cube.py
# There may be more data points available for non-exceptional cf IW_analysis.py
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
              
# 32^3 x 64
plistIWf = [(-3,0,3,0), (-4,0,4,0), (-4,0,4,0),
            (-5,0,5,0), (-5,0,5,0)]
twlistIWf = [0.25, -0.75, 0.375, -0.625, 0.375]
gflist004 = [1700, 1740, 1780, 1820, 1860, 1900, 1940, 1980,
             2020, 2060, 2100, 2140, 2180, 2220, 2260]
gflist006 = [1000, 1040, 1080, 1120, 1160, 1200, 1240, 1280,
             1320, 1360, 1400, 1440, 1480]
gflist008 = [1800, 1840, 1880, 1920, 1960, 2000, 2040, 2080,
             2120, 2160]
             
def load_IWc_Exceptional_Data(m, plist, twlist, gflist):
    return [npr.IWc_Exceptional_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]

def load_IWf_Exceptional_Data(m, plist, twlist, gflist):
    return [npr.IWf_Exceptional_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]
            
def load_IWf_Data(m, plist, twlist, gflist):
    return [npr.IWf_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]

def load_IWc_Data(m, plist, twlist, gflist):
    return [npr.IWc_Data(m, p, tw, gflist)
            for (p, tw) in zip(plist, twlist)]
            
def calc_Zs_Exceptional(Data):
    Data.load()
    me.bilinear_Lambdas(Data)
    me.bilinear_LambdaJK(Data)
    #m.fourquark_Zs(Data)
    #m.fourquark_ZsJK(Data)
    Data.clear()
    return Data
              
def calc_Zs(Data):
    Data.load()
    m.bilinear_Lambdas(Data)
    m.bilinear_LambdaJK(Data)
    #m.fourquark_Zs(Data)
    #m.fourquark_ZsJK(Data)
    Data.clear()
    return Data
    
def main():

    print 'Loading data...'
    data_03 = load_IWc_Data(.03, plist_IWc, twlist_IWc, gflist_03)
    data_02 = load_IWc_Data(.02, plist_IWc, twlist_IWc, gflist_02)
    data_01 = load_IWc_Data(.01, plist_IWc, twlist_IWc, gflist_01)
    data_005 = load_IWc_Data(.005, plist_IWc, twlist_IWc, gflist_005)
    
    data_004 = load_IWf_Data(.004, plistIWf, twlistIWf, gflist004)
    data_006 = load_IWf_Data(.006, plistIWf, twlistIWf, gflist006)
    data_008 = load_IWf_Data(.008, plistIWf, twlistIWf, gflist008)
    
#    dataE_03 = load_IWc_Exceptional_Data(.03, plist_IWc, twlist_IWc, gflist_03)
#    dataE_02 = load_IWc_Exceptional_Data(.02, plist_IWc, twlist_IWc, gflist_02)
#    dataE_01 = load_IWc_Exceptional_Data(.01, plist_IWc, twlist_IWc, gflist_01)
#    dataE_005 = load_IWc_Exceptional_Data(.005, plist_IWc, twlist_IWc, gflist_005)
#    
#    dataE_004 = load_IWf_Exceptional_Data(.004, plistIWf, twlistIWf, gflist004)
#    dataE_006 = load_IWf_Exceptional_Data(.006, plistIWf, twlistIWf, gflist006)
#    dataE_008 = load_IWf_Exceptional_Data(.008, plistIWf, twlistIWf, gflist008)

    print [d.mu for d in data_005]
    print [d.mu for d in data_004]
        
    print 'Computing stuff..'
    pool = Pool()
    
#    data_03 = pool.map_async(calc_Zs, data_03).get()
    data_02 = pool.map_async(calc_Zs, data_02).get()
    data_01 = pool.map_async(calc_Zs, data_01).get()   
    data_005 = pool.map_async(calc_Zs, data_005).get()

    data_004 = pool.map_async(calc_Zs, data_004).get()
    data_006 = pool.map_async(calc_Zs, data_006).get()
    data_008 = pool.map_async(calc_Zs, data_008).get()
#    
#    dataE_03 = pool.map_async(calc_Zs_Exceptional, dataE_03).get()
#    dataE_02 = pool.map_async(calc_Zs_Exceptional, dataE_02).get()
#    dataE_01 = pool.map_async(calc_Zs_Exceptional, dataE_01).get()   
#    dataE_005 = pool.map_async(calc_Zs_Exceptional, dataE_005).get()

#    dataE_004 = pool.map_async(calc_Zs_Exceptional, dataE_004).get()
#    dataE_006 = pool.map_async(calc_Zs_Exceptional, dataE_006).get()
#    dataE_008 = pool.map_async(calc_Zs_Exceptional, dataE_008).get()

    pool.close()
    pool.join()
    
#    #print [(1/d.Z_S['g']) for d in data_005]
#    print match.Cm(match.alpha_s2, 'g')
#    print match.Cm(match.alpha_s2, 'q')
#    print match.Cm(match.alpha_s3, 'g')
#    print match.Cm(match.alpha_s3, 'q')

#    print match.Cm(match.alpha_s2, 'g')/(data_005[0].ZA*data_005[0].Z_S['g'])
#    print match.Cm(match.alpha_s3, 'g')/(data_005[0].ZA*data_005[-1].Z_S['g'])
#    print match.Cm(match.alpha_s2, 'q')/(data_005[0].ZA*data_005[0].Z_S['q'])
#    print match.Cm(match.alpha_s3, 'q')/(data_005[0].ZA*data_005[-1].Z_S['q'])
    ZAf = npr.IWf_Data.ZA
    ZAc = npr.IWc_Data.ZA
    print data_02[0].Z_S['g'], data_02[0].Z_S_sigmaJK['g']
    print data_01[0].Z_S['g'], data_01[0].Z_S_sigmaJK['g']
    print data_005[0].Z_S['g'], data_005[0].Z_S_sigmaJK['g']
    
    data_0c = map(fits.line_fit_bilinears, data_005, data_01, data_02)
    print data_0c[0].Z_S['g'], data_0c[0].Z_S_sigmaJK['g']
    
    print ''
    
    print data_008[0].Z_S['g'], data_008[0].Z_S_sigmaJK['g']
    print data_006[0].Z_S['g'], data_006[0].Z_S_sigmaJK['g']
    print data_004[0].Z_S['g'], data_004[0].Z_S_sigmaJK['g']
    
    data_0f = map(fits.line_fit_bilinears, data_004, data_006, data_008)
    print data_0f[0].Z_S['g'], data_0f[0].Z_S_sigmaJK['g']
    print '\n Z_S'
    print '32^3 lattice'
    for scheme in 'g', 'q':
        print 'scheme', scheme
        cfac = C_S(alpha_s2, scheme)*ZAf
        Z = data_0f[0].Z_S[scheme]
        sig = data_0f[0].Z_S_sigmaJK[scheme]
        print ' 2 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)
        cfac = C_S(alpha_s3, scheme)*ZAf
        Z = data_0f[-2].Z_S[scheme]
        sig = data_0f[-2].Z_S_sigmaJK[scheme]
        print ' 3 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)
        print ''
        
    print '24^3 lattice'
    for scheme in 'g', 'q':
        print 'scheme', scheme
        cfac = C_S(alpha_s2, scheme)*ZAc
        Z = data_0c[0].Z_S[scheme]
        sig = data_0c[0].Z_S_sigmaJK[scheme]
        print ' 2 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)
        cfac = C_S(alpha_s3, scheme)*ZAc
        # Small interpolation in mu required.
        d1 = data_0c[-2]
        d2 = data_0c[-1]
        #print d1.mu, d1.Z_S[scheme]
        #print d2.mu, d2.Z_S[scheme]
        fit = fits.line_fit2([(d1.mu, d1.Z_S[scheme], d1.Z_S_sigmaJK[scheme]), 
                              (d2.mu, d2.Z_S[scheme], d2.Z_S_sigmaJK[scheme])])
            
        Z = fit.b*3 + fit.a
        #print Z
        sig = max(d1.Z_S_sigmaJK[scheme], d2.Z_S_sigmaJK[scheme]) # Fudge.
        print ' 3 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)

    print '\n\n Z_P'
    print '32^3 lattice'
    for scheme in 'g', 'q':
        print 'scheme', scheme
        cfac = C_S(alpha_s2, scheme)*ZAf
        Z = data_0f[0].Z_P[scheme]
        sig = data_0f[0].Z_P_sigmaJK[scheme]
        print ' 2 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)
        cfac = C_S(alpha_s3, scheme)*ZAf
        Z = data_0f[-2].Z_P[scheme]
        sig = data_0f[-2].Z_P_sigmaJK[scheme]
        print ' 3 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)
        print ''
        
    print '24^3 lattice'
    for scheme in 'g', 'q':
        print 'scheme', scheme
        cfac = C_S(alpha_s2, scheme)*ZAc
        Z = data_0c[0].Z_P[scheme]
        sig = data_0c[0].Z_P_sigmaJK[scheme]
        print ' 2 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)
        cfac = C_S(alpha_s3, scheme)*ZAc
        # Small interpolation in mu required.
        d1 = data_0c[-2]
        d2 = data_0c[-1]
        #print d1.mu, d1.Z_P[scheme]
        #print d2.mu, d2.Z_P[scheme]
        fit = fits.line_fit2([(d1.mu, d1.Z_P[scheme], d1.Z_P_sigmaJK[scheme]), 
                              (d2.mu, d2.Z_P[scheme], d2.Z_P_sigmaJK[scheme])])
            
        Z = fit.b*3 + fit.a
        #print Z
        sig = max(d1.Z_P_sigmaJK[scheme], d2.Z_P_sigmaJK[scheme]) # Fudge.
        print ' 3 GeV: {0:.4f} +/- {1:.4f}'.format(cfac*Z, cfac*sig)
        
    return 0
    
if __name__ == "__main__":
    sys.exit(main())
