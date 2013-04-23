import os, sys
import numpy as np

from math import sqrt

def to_numpy(lines):
    '''Remove the leading column; 
       convert the remaining numbers into an array.'''
    tmp = [map(float, line.split()[1:]) for line in lines]
    return np.array(tmp)

def convert_file(file):
    '''Convert tabular text file into a numpy array.'''
    with open(file) as f:
        return to_numpy(f.readlines())
        
def extract(base, dirs):
    '''Extract data from base file in each directory.
       Return a list of the results.'''

    results = []
    filename = lambda d, base: './{0}/traj_{0}_{1}.txt'.format(d, base)
    for d in dirs:
        f = filename(d, base)
        print f
        try:
            results.append(convert_file(f))
        except IOError as e:
            print e
            continue
            
    return results

def averages_JK(list_np):
    '''Average and jackknife averages of numpy arrays.'''
    L = len(list_np)
    ave = sum(list_np)/L
    result = [ave]
    for x in range(L):
        tmp = list_np.pop(0)  # Remove first element.
        result.append(sum(list_np)/(L-1))
        list_np.append(tmp)
    assert len(result) == L+1
    return result

def output(file, list_np):
    '''Output data in tabular format.'''
    if type(list_np) is list:
        list_np = np.hstack(list_np)
    np.savetxt(file, list_np, fmt='%.12e', delimiter='    ')
    
def block_ave(arr, T=64):
    '''Average every Tth row.  Both source and sink are varied over the entire
       lattice, this averages rows with the same source-sink separation.'''
    return sum(np.vsplit(arr, T))/T  
       
def get_real(arr):
    '''Return the real number columns (0, 2, 4, ...).'''
    return arr[:,::2]

def pioncorr_p0(dirs):
    '''Pion correlator.'''
    base = 'pioncorr_p0'
    cs = extract(base, dirs)
    cs = map(get_real, cs)
    cs = map(block_ave, cs)
    aves = averages_JK(cs)
    output(base+'.txt', aves)
    return aves

def kaoncorr_s0_p0(dirs):
    '''Kaon correlator.'''
    base = 'kaoncorr_s0_p0'
    cs = extract(base, dirs)
    cs = map(get_real, cs) 
    cs = map(block_ave, cs)
    aves = averages_JK(cs)
    output(base+'.txt', aves)
    return aves

def vacuum_corr(bubbles):
    '''Input: Gauge list of columns Q(t)=<L(t) L(t)^dag>.
       Output: Ave, jksamples of (1/T)*sum_t'[Q(t') Q(t+t')]'''
    T = len(bubbles[0])
    aves = averages_JK(bubbles)
    N = len(aves)
    aves = np.hstack(aves)  
    reduce = lambda t: np.average(aves*np.roll(aves, -t, axis=0), axis=0)
    result = np.vstack([reduce(t) for t in range(T)])
    return np.hsplit(result, N)
    
def np_to_list(arr):
    '''List of columns of numpy array.'''
    N = arr.shape[1]  # Number of columns.
    return np.hsplit(arr, N)

    
def pipicorr(sep, dirs):
    '''pipi correlators.'''
    
    def figureC_hack(sep, dirs):
        '''This is to correct the typeC contractions on the 26 corrupted
           configurations.  The files Daiqian has provided already sum
           over the 64 blocks so they need to be treated separately.'''
           
        corrupt = [4080, 4120, 4160, 4200, 4240, 4280, 4320, 4360, 4400,
                   4440, 4480, 4520, 4560, 4600, 4640, 4680, 4720, 4760,
                   4800, 4840, 4880, 4920, 4960, 5000, 5040, 5080]
        assert len(corrupt)==26
        index = dirs.index('4080')
        
        Cs = extract(Cbase, dirs)
        Cs = np.array(map(block_ave, Cs))
        
        # Remove corrupted entries.
        Cs[index:index+26, :, :] = 0
        #print Cs[index]
        
        # Replace with correct values.
        filename = lambda d, base: './figureC_hack/{0}/traj_{0}_{1}.txt'.format(d, base)
        new = []
        for d in corrupt:
            f = filename(d, Cbase)
            print f
            try:
                new.append(convert_file(f))
            except IOError as e:
                print e
                continue
        Cs[index:index+26, :, :] = np.array(new)
        #print Cs[index]
        
        return Cs
        
    
    # Process data.
    Cbase = 'figureC0_sep{0}'.format(sep)  # Cross.
    Dbase = 'figureD0_sep{0}'.format(sep)  # Direct.
    Rbase = 'figureR0_sep{0}'.format(sep)  # Rectangular.
    Vbase = 'figureV0_sep{0}'.format(sep)  # Vacuum.
    subbase = 'dis_V0_sep{0}'.format(sep)  # Bubble.
    print 'Calculating sep{0} pion-pion correlation functions:'.format(sep)
    #Cs = extract(Cbase, dirs)
    Cs = figureC_hack(sep, dirs) # figureC hack.
    Ds = extract(Dbase, dirs)
    Rs = extract(Rbase, dirs)
    Vs = extract(Vbase, dirs)
    subs = extract(subbase, dirs)
    #Cs = np.array(map(block_ave, Cs))
    Ds = np.array(map(block_ave, Ds))
    Rs = np.array(map(block_ave, Rs))
    Vs = np.array(map(block_ave, Vs))
        
    # Construct correlators.
    I0Vs = 2*Ds - 6*Rs + Cs  # Vacuum ignored.
    I2s = 2*Ds - 2*Cs
    
    I0V = averages_JK(list(I0Vs))  # -Revert to list.
    I2 = averages_JK(list(I2s))
    V = averages_JK(list(Vs))
    Vsub = vacuum_corr(subs) 
    I0 = [i0v + 3*(v-vsub) for i0v,v,vsub in zip(I0V,V,Vsub)]
    
    # Extract the real number columns.
    I0V = map(get_real, I0V)
    I2 = map(get_real, I2)
    I0 = map(get_real, I0)
    
    # Rotate data by 'sep' for fitting.
    I0V = np.roll(np.hstack(I0V), sep, axis=0)
    I2 = np.roll(np.hstack(I2), sep, axis=0)
    I0 = np.roll(np.hstack(I0), sep, axis=0)
    
    # Output.
    output('pipi_I0V_sep{0}.txt'.format(sep), I0V)
    output('pipi_I2_sep{0}.txt'.format(sep), I2)
    output('pipi_I0_sep{0}.txt'.format(sep), I0)
    
def assert_identities():
    pass
    
def S5D(sep, dt, dirs):
    base3 = 'type3S5D_sep_{0}_s_0_deltat_{1}'.format(sep, dt)
    base4 = 'type4S5D_sep_{0}_s_0_deltat_{1}'.format(sep, dt)
    t3 = extract(base3, dirs)
    t4 = extract(base4, dirs)
    t3 = map(block_ave, t3)
    t4 = map(block_ave, t4)
    
    # Extract appropriate columns.
    t3 = np.hstack(averages_JK([c[:,:2] for c in t3]))
    t4 = np.hstack(averages_JK([c[:,:2] for c in t4]))
    
    # Extract real number columns.
    t3 = get_real(t3)
    t4 = get_real(t4)
    
    return t3, t4

def Kpipi_I2(sep, dt, dirs):
    '''<(pi pi)_{I=2}| Q_{1,2,7-10} |K>.'''
    
    # Extract data.
    base = 'type1_sep_{0}_s_0_deltat_{1}'.format(sep, dt)
    cs = extract(base, dirs)
    N = len(cs)  # Number of configurations.
    cs = map(block_ave, cs)
    
    # Extract by contraction.
    contractions = []
    get_columns = lambda c,i: c[:,4*i:4*i+2]
    for i in range(8):
        contractions.append([get_columns(c,i) for c in cs])
        
    # Now have 8 sets of data; one for each contraction.
    # Compute jackknife samples.
    cc = [np.hstack(averages_JK(contraction)) for contraction in contractions]

    # Extract real number columns.
    cc = map(get_real, cc)
    c = lambda i:cc[i-1]  # Note index change.

    # Output contractions 1 and 2.
    for n in range(1,11):
        output('kpipi_c{0}_sep{1}_dt{2}.txt'.format(n, sep, dt), c(n))
    
    # Construct operators.
    A2 = [0]*10
    A2[0] = sqrt(2./3)*(c(1) - c(5))  
    A2[1] = sqrt(2./3)*(c(2) - c(6))
    # A2[2-5] = 0.
    A2[6] = sqrt(3./2)*(c(3) - c(7))
    A2[7] = sqrt(3./2)*(c(4) - c(8))
    A2[8] = sqrt(3./2)*(c(1) - c(5))
    A2[9] = sqrt(3./2)*(c(2) - c(6))
    
    # Output correlation functions.
    for i in 0,1,6,7,8,9:
        output('kpipi_I2_Q{0}_sep{1}_dt{2}.txt'.format(i+1, sep, dt), A2[i])
        
def contractions(sep, dt, dirs):
    '''Individual contractions and subtractions.'''
    # Extract data.
    Tbase = ['type{0}_sep_{1}_s_0_deltat_{2}'.format(T, sep, dt) 
             for T in (1,2,3,4)]
    Ts = [extract(Tbase[T], dirs) for T in range(4)]  # Note index change.
    N = len(Ts[0])  # Number of configurations.
    process = lambda c: sum(np.vsplit(c, 64))/64
    get_columns = lambda c,i: c[:,4*i:4*i+2]
    Ts = [map(process, T) for T in Ts]  # Average over shifts.
    
    # Extract by contraction.
    contractions=[]
    for T in Ts[0], Ts[1]:  # Type 1 and 2 contractions.
        for i in range(8):
            contractions.append([get_columns(c,i) for c in T])
            
    for T in Ts[2], Ts[3]:  # Type 3 and 4 contractions.
        for i in range(16):
            contractions.append([get_columns(c,i) for c in T])
    
    # Now have 48 sets of data; one for each contraction.
    # Compute jackknife samples.
    cc = [np.hstack(averages_JK(contraction)) for contraction in contractions]
    
    # Extract real number columns.
    cc = map(get_real, cc)

    # Construct operators.
    c = lambda i:cc[i-1]
    
    alpha = K_to_vac(sep, dt, dirs)
    mix3, mix4 = S5D(sep, dt, dirs)
    
    for i in range(1,49):
        output('kpipi_c{0}_sep{1}_dt{2}.txt'.format(i, sep, dt), c(i))
    for i in range(10):
        output('kpipi_Q{0}mix3_sep{1}_dt{2}.txt'.format(i+1, sep, dt),
                alpha[i]*mix3)
        output('kpipi_Q{0}mix4_sep{1}_dt{2}.txt'.format(i+1, sep, dt),
                alpha[i]*mix4)
                                                           
            
def K_to_vac(sep, dt, dirs):
    '''<0|Q_{1-10}|K>/<0|s5d|K>.  These diagrams are all of type4.'''   
    
    # Extract data.
    Qbase = 'type4_sep_{0}_s_0_deltat_{1}'.format(sep, dt)
    S5Dbase = 'type4S5D_sep_{0}_s_0_deltat_{1}'.format(sep, dt)
    cs = extract(Qbase, dirs)
    cs5 = extract(S5Dbase, dirs)
    cs = map(block_ave, cs)
    cs5 = map(block_ave, cs5)
    
    # K to vacuum contractions.
    contractions = []
    get_columns = lambda c,i: c[:,4*i+2:4*i+4]
    for i in range(16):
        contractions.append([get_columns(c,i) for c in cs])
    cs5 = np.hstack(averages_JK([c[:,2:] for c in cs5]))
        
    # Now have 16 sets of data; one for each contraction.
    # Compute jackknife samples
    cc = [np.hstack(averages_JK(contraction)) for contraction in contractions]
    
    # Extract real number columns.
    cc = map(get_real, cc)
    cs5 = get_real(cs5)
        
    # Construct K -> vacuum amplitudes.  Note index change.
    c = lambda i:cc[i-32-1]
    
    S = [0]*10
    S[0] = sqrt(1./3)*(-3*c(33))
    
    S[1] = sqrt(1./3)*(-3*c(34))
    
    S[2] = sqrt(3.)*(-2*c(33) - c(37) + c(41) + c(45))
                       
    S[3] = sqrt(3.)*(-2*c(34) - c(38) + c(42) + c(46))
                       
    S[4] = sqrt(3.)*(-2*c(35) - c(39) + c(43) + c(47))
                       
    S[5] = sqrt(3.)*(-2*c(36) - c(40) + c(44) + c(48))
                       
    S[6] = (sqrt(3.)/2)*(-c(35) + c(39) - c(43) - c(47))
    
    S[7] = (sqrt(3.)/2)*(-c(36) + c(40) - c(44) - c(48)) 
                          
    S[8] = (sqrt(3.)/2)*(-c(33) + c(37) - c(41) - c(45))
    
    S[9] = (sqrt(3.)/2)*(-c(34) + c(38) - c(42) - c(46))
    
    alpha = [s/cs5 for s in S]
    return alpha
    '''
    for i in range(10):
        output('kpipi_alpha_Q{0}_sep{1}_dt{2}.txt'.format(i+1, sep, dt), 
                alpha[i])'''
    
def Kpipi_I0(sep, dt, dirs):
    '''<(pi pi)_{I=0}| Q_{1-10} |K>.'''
      
    # Extract data.
    Tbase = ['type{0}_sep_{1}_s_0_deltat_{2}'.format(T, sep, dt) 
             for T in (1,2,3,4)]
    Ts = [extract(Tbase[T], dirs) for T in range(4)]  # Note index change.
    N = len(Ts[0])  # Number of configurations.
    process = lambda c: sum(np.vsplit(c, 64))/64
    get_columns = lambda c,i: c[:,4*i:4*i+2]
    Ts = [map(process, T) for T in Ts]  # Average over shifts.
    
    # Extract by contraction.
    contractions=[]
    for T in Ts[0], Ts[1]:  # Type 1 and 2 contractions.
        for i in range(8):
            contractions.append([get_columns(c,i) for c in T])
            
    for T in Ts[2], Ts[3]:  # Type 3 and 4 contractions.
        for i in range(16):
            contractions.append([get_columns(c,i) for c in T])
    
    # Now have 48 sets of data; one for each contraction.
    # Compute jackknife samples.
    cc = [np.hstack(averages_JK(contraction)) for contraction in contractions]
    
    # Extract real number columns.
    cc = map(get_real, cc)

    # Construct operators.
    c = lambda i:cc[i-1]
    
    A0 = [0]*10
    A0[0] = sqrt(1./3)*(-c(1) - 2*c(5) + 3*c(9) + 3*c(17) - 3*c(33))
    
    A0[1] = sqrt(1./3)*(-c(2) - 2*c(6) + 3*c(10) + 3*c(18) - 3*c(34))
    
    A0[2] = sqrt(3.)*(-c(5) + 2*c(9) - c(13) + 2*c(17) + c(21) -
                       c(25) - c(29) - 2*c(33) - c(37) + c(41) + c(45))
                       
    A0[3] = sqrt(3.)*(-c(6) + 2*c(10) - c(14) + 2*c(18) + c(22) -
                       c(26) - c(30) - 2*c(34) - c(38) + c(42) + c(46))
                       
    A0[4] = sqrt(3.)*(-c(7) + 2*c(11) - c(15) + 2*c(19) + c(23) -
                       c(27) - c(31) -2*c(35) - c(39) + c(43) + c(47))
                       
    A0[5] = sqrt(3.)*(-c(8) + 2*c(12) - c(16) + 2*c(20) + c(24) -
                       c(28) - c(32) -2*c(36) - c(40) + c(44) + c(48))
                       
    A0[6] = (sqrt(3.)/2)*(-c(3) - c(7) + c(11) + c(15) + c(19) -
                          c(23) + c(27) + c(31) - c(35) + c(39) - c(43) - c(47))
    
    A0[7] = (sqrt(3.)/2)*(-c(4) - c(8) + c(12) + c(16) + c(20) -
                          c(24) + c(28) + c(32) - c(36) + c(40) - c(44) - c(48)) 
                          
    A0[8] = (sqrt(3.)/2)*(-c(1) - c(5) + c(9) + c(13) + c(17) -
                          c(21) + c(25) + c(29) - c(33) + c(37) - c(41) - c(45))
    
    A0[9] = (sqrt(3.)/2)*(-c(2) - c(6) + c(10) + c(14) + c(18) -
                          c(22) + c(26) + c(30) - c(34) + c(38) - c(42) - c(46))
                         
    # Subtractions.
    alpha = K_to_vac(sep, dt, dirs)
    mix3, mix4 = S5D(sep, dt, dirs)
    
    A0_sub = [A - a*(-mix3+mix4) for (A, a) in zip(A0, alpha)]
    
    # Output correlation functions.
    for i in range(10):
        output('kpipi_I0_Q{0}_sep{1}_dt{2}.txt'.format(i+1, sep, dt), 
                A0_sub[i])
                        
def Kpipi_I0_connected(sep, dt, dirs):
    '''<(pi pi)_{I=0}| Q_{1-10} |K>, neglecting disconnected diagrams.'''
      
    # Extract data.
    Tbase = ['type{0}_sep_{1}_s_0_deltat_{2}'.format(T, sep, dt) 
             for T in (1,2,3,4)]
    Ts = [extract(Tbase[T], dirs) for T in range(3)]  # Note index change.
    N = len(Ts[0])  # Number of configurations.
    process = lambda c: sum(np.vsplit(c, 64))/64
    get_columns = lambda c,i: c[:,4*i:4*i+2]
    Ts = [map(process, T) for T in Ts]  # Average over shifts.
    
    # Extract by contraction.
    contractions=[]
    for T in Ts[0], Ts[1]:  # Type 1 and 2 contractions.
        for i in range(8):
            contractions.append([get_columns(c,i) for c in T])
            
    # Type 3 contractions.  Type 4 are disconnected and thus excluded.
    for i in range(16):
        contractions.append([get_columns(c,i) for c in Ts[2]])
    
    # Now have 32 sets of data; one for each contraction.
    # Compute jackknife samples.
    cc = [np.hstack(averages_JK(contraction)) for contraction in contractions]
    
    # Extract real number columns.
    cc = map(get_real, cc)

    # Construct operators.
    c = lambda i:cc[i-1]
    
    A0 = [0]*10
    A0[0] = sqrt(1./3)*(-c(1) - 2*c(5) + 3*c(9) + 3*c(17))
    
    A0[1] = sqrt(1./3)*(-c(2) - 2*c(6) + 3*c(10) + 3*c(18))
    
    A0[2] = sqrt(3.)*(-c(5) + 2*c(9) - c(13) + 2*c(17) + c(21) -
                       c(25) - c(29))
                       
    A0[3] = sqrt(3.)*(-c(6) + 2*c(10) - c(14) + 2*c(18) + c(22) -
                       c(26) - c(30))
                       
    A0[4] = sqrt(3.)*(-c(7) + 2*c(11) - c(15) + 2*c(19) + c(23) -
                       c(27) - c(31))
                       
    A0[5] = sqrt(3.)*(-c(8) + 2*c(12) - c(16) + 2*c(20) + c(24) -
                       c(28) - c(32))
                       
    A0[6] = (sqrt(3.)/2)*(-c(3) - c(7) + c(11) + c(15) + c(19) -
                          c(23) + c(27) + c(31))
    
    A0[7] = (sqrt(3.)/2)*(-c(4) - c(8) + c(12) + c(16) + c(20) -
                          c(24) + c(28) + c(32)) 
                          
    A0[8] = (sqrt(3.)/2)*(-c(1) - c(5) + c(9) + c(13) + c(17) -
                          c(21) + c(25) + c(29))
    
    A0[9] = (sqrt(3.)/2)*(-c(2) - c(6) + c(10) + c(14) + c(18) -
                          c(22) + c(26) + c(30))
                          
    # Subtractions.  "mix4" is disconnected and thus excluded.
    alpha = K_to_vac(sep, dt, dirs)
    mix3 = S5D(sep, dt, dirs)[0]
    
    A0_sub = [A - a*(-mix3) for (A, a) in zip(A0, alpha)]
    
    # Output correlation functions.
    for i in range(10):
        output('kpipi_I0_connected_Q{0}_sep{1}_dt{2}.txt'.format(i+1, sep, dt), 
                A0_sub[i])
        
def main(argv):
    #pioncorr_p0(argv)        
    #kaoncorr_s0_p0(argv)
    pipicorr(0, argv)
    pipicorr(2, argv)
    pipicorr(4, argv)
    #for sep in 0,2,4:
    #    for dt in 12,16,20,24,28:
    #        Kpipi_I2(sep, dt, argv)
    #S5D(4, 16, argv)
    #K_to_vac(4, 16, argv)
    #Kpipi_I2(4, 24, argv)
    #Kpipi_I0(4, 24, argv)
    #Kpipi_I0_connected(4, 20, argv)
    #K_to_vac(4, 16, argv)
    #contractions(4, 16, argv)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
