import sys, numpy
from numpy import dot
from numpy.linalg import inv
from multiprocessing import Pool

import fits
import pyNPR as npr
import measurements as m

numpy.set_printoptions(precision=5, suppress=True)

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

plistIWf_a = plistIWf_b = [(-2, 0, 2, 0)]
twlistIWf_a, twlistIWf_b = [-.413], [.783]

gflist004_a = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
              1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650,
              1700, 1750, 1800, 1850, 1900, 1950]
gflist004_b = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
              1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]
gflist006_a = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650,
               1700, 1750]
gflist006_b = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]
gflist008_ = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
              1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650]

plist3IWf = [(-5,0,5,0)]
twlist3IWf = [-0.531292]
gflist004_3 = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
gflist006_3 = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300, 1350]
gflist008_3 = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250,
               1300]

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
gflist01_ = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200] #, 2300]
gflist02_ = [700, 750, 800, 850, 900, 950, 1000, 1050]

plist3IWc = [(-5,0,5,0)]
twlist3IWc = [-0.632547]
gflist005_3 = [900, 950, 1000, 1050, 1100, 1150, 1200, 1250]
gflist01_3 = [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200]
gflist02_3 = [700, 750, 800, 850, 900, 950, 1000, 1050]

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
    
def main():
    print "Initializing data structures...",
    # Fine IW.
    data004 = load_IWf_Data(.004, plistIWf_a, twlistIWf_a, gflist004_a) +\
              load_IWf_Data(.004, plist3IWf, twlist3IWf, gflist004_3)

    data006 = load_IWf_Data(.006, plistIWf_a, twlistIWf_a, gflist006_a) +\
              load_IWf_Data(.006, plist3IWf, twlist3IWf, gflist006_3)

    data008 = load_IWf_Data(.008, plistIWf_a, twlistIWf_a, gflist008_) +\
              load_IWf_Data(.008, plist3IWf, twlist3IWf, gflist008_3)

    # Coarse IW.
    data005 = load_IWc_Data(.005, plistIWc_, twlistIWc_, gflist005_) +\
              load_IWc_Data(.005, plist3IWc, twlist3IWc, gflist005_3)

    data01 = load_IWc_Data(.01, plistIWc_, twlistIWc_, gflist01_) +\
             load_IWc_Data(.01, plist3IWc, twlist3IWc, gflist01_3)

    data02 = load_IWc_Data(.02, plistIWc_, twlistIWc_, gflist02_) +\
             load_IWc_Data(.02, plist3IWc, twlist3IWc, gflist02_3)
    
    print "complete"
    
    print "Computing Zs...",
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
    print "complete"

    # Chiral Limits.
    data0c = map(fits.line_fit_Data, data005, data01, data02)
    data0f = map(fits.line_fit_Data, data004, data006, data008)
    
    M = numpy.array([[1.,0,0],[0,2.,0],[0,0,-4.]])  # Change to delta S=1 basis. 
    
    print [d.mu for d in data0c]
    print [d.mu for d in data0f]
    
    d1 = data0c[0]
    print 'coarse', d1.mu, 'GeV'
    print 'g-scheme:'
    print d1.thing_chi, '\n'
    print d1.thing_chi_sigma, '\n'
    #print dot(dot(M,thing), inv(M))
    print 'q-scheme:'
    print d1.thing_q_chi, '\n'
    print d1.thing_q_chi_sigma, '\n'
    
    
    d2 = data0f[0]
    print 'fine', d2.mu, 'GeV'

    print 'g-scheme:'
    print d2.thing_chi, '\n'
    print d2.thing_chi_sigma, '\n'
    #print dot(dot(M,thing), inv(M))
    print 'q-scheme:'
    print d2.thing_q_chi, '\n'
    print d2.thing_q_chi_sigma, '\n' 
    '''   
    d3 = data0c[-1]
    print 'coarse', d3.mu, 'GeV'
    print 'g-sheme:'
    print d3.thing_chi, '\n'
    print d3.thing_chi_sigma, '\n'
    #print dot(dot(M,thing), inv(M))
    print 'q-scheme:'
    print d3.thing_q_chi, '\n'
    print d3.thing_q_chi_sigma, '\n'
       
    d4 = data0f[-1]
    print 'fine', d4.mu, 'GeV'

    #print d4.thing_chi, '\n'
    #print d4.thing_chi_sigma, '\n'
    #print dot(dot(M,thing), inv(M))
    print 'q-scheme:'
    print d4.thing_q_chi, '\n'
    print d4.thing_q_chi_sigma, '\n'    
    '''
    return 0

if __name__ == "__main__":
    sys.exit(main())
