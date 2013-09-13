import numpy as np
from numpy import array, trace, dot, tensordot
from numpy.linalg import inv

import domain_wall as dw
from domain_wall import Gc
from pyNPR import sgn


# Some stats stuff
def JKsample(list):
    '''Yield the jackknife samples of the elements in list.'''
    sample = []
    for dummy in range(len(list)):
        x, xs = list[0], list[1:]
        sample.append(xs[:])
        xs.append(x)
        list = xs
    return sample

def JKsigma(JKvals, ave):
    N = len(JKvals)
    diffs = [(JKval - ave)*(JKval - ave) for JKval in JKvals]
    return np.sqrt(sum(diffs)*(1-1./N))

def bootstrap_sample(data, N):
    '''Yield N boostrap samples of the elements in data.'''
    ri = random.randint
    L = len(data)
    sample = []
    for foo in range(N):
        sample.append([data[ri(0, L-1)] for bar in range(L)])
    return sample


# Measurements

def amputate_bilinears(inprop_list, outprop_list, bilinear_array):
    N = len(inprop_list)
    assert N == len(outprop_list) == len(bilinear_array)
    inprop_inv = inv(sum(inprop_list)/N)
    outprop_inv = inv(sum(outprop_list)/N)
    bilinears = sum(bilinear_array)/N
    return [reduce(dot, [outprop_inv, bilinears[g], inprop_inv])
            for g in range(16)]

def bilinear_Lambdas(Data):
    # type check? data loaded?
    amputated = amputate_bilinears(Data.inprop_list,
                                    Data.outprop_list,
                                    Data.bilinear_array)
    norm = Data.V/12.
    Lambda = lambda x, y: (trace(dot(dw.hc(x), y)).real)*norm
    Data.Lambda = map(Lambda, Gc, amputated)

    # For the (X, g) schemes.
    Data.Lambda_V = (Data.Lambda[1] + Data.Lambda[2] + Data.Lambda[4] +
                     Data.Lambda[8])*(1./4)   
    Data.Lambda_A = (Data.Lambda[14] + Data.Lambda[13] + Data.Lambda[11] +
                     Data.Lambda[7])*(1./4)
    Data.Lambda_VpA = (Data.Lambda[1] + Data.Lambda[2] + Data.Lambda[4] +
                       Data.Lambda[8] + Data.Lambda[14] + Data.Lambda[13] +
                       Data.Lambda[11] + Data.Lambda[7])*(1./8)
    Data.Lambda_VmA = 2*(Data.Lambda_V - Data.Lambda_A)/\
                        (Data.Lambda_V + Data.Lambda_A)
    #Data.Lambda_VmA = (Data.Lambda_V - Data.Lambda_A)
    Data.Lambda_PmS = 2*(Data.Lambda[15] - Data.Lambda[0])/\
                        (Data.Lambda[15] + Data.Lambda[0])
    # For the (X, q) schemes.
    aq = Data.aq
    Gmu = (amputated[1], amputated[2], amputated[4], amputated[8])
    qmuGmu = sum([aq[i]*Gmu[i] for i in range(4)])
    Data.Vq = (trace(dot(qmuGmu, dw.slash(aq))).real)*norm/Data.apSq
    Gmu5 = (amputated[14], -amputated[13], amputated[11], -amputated[7])
    qmuGmu5 = sum([aq[i]*Gmu5[i] for i in range(4)])
    tmp = reduce(dot, [qmuGmu5, Gc[15], dw.slash(aq)]).real
    Data.Vq5 = trace(tmp*norm)/Data.apSq
    Data.Vq = (Data.Vq + Data.Vq5)/2 #fix
    
    # Z-factors.
    def Z_S(Lambda, Lambda_vec):
        'Lambda_vec depends on how Zq is determined (see usage below).'
        return Lambda_vec/Lambda[0]
        
    Data.Z_S = dict(g=None, q=None)
    Data.Z_S['g'] = Z_S(Data.Lambda, Data.Lambda_VpA)
    Data.Z_S['q'] = Z_S(Data.Lambda, Data.Vq)

def bilinear_LambdaJK(Data):
    amputatedJK = map(amputate_bilinears, JKsample(Data.inprop_list),
                                          JKsample(Data.outprop_list),
                                          JKsample(Data.bilinear_array))
    norm = Data.V/12.
    Lambda = lambda x, y: (trace(dot(dw.hc(x), y)).real)*norm
    Data.LambdaJK = [np.array(map(Lambda, Gc, amp)) for amp in amputatedJK]
    Data.Lambda_sigmaJK = JKsigma(Data.LambdaJK, Data.Lambda)
    
    # For the (X, g) schemes.
    def V(Lambda):
        return (Lambda[1] + Lambda[2] + Lambda[4] + Lambda[8])*(1/4.)
    def A(Lambda):
        return (Lambda[14] + Lambda[13] + Lambda[11] + Lambda[7])*(1/4.)
    def VpA(Lambda):
        return (Lambda[1] + Lambda[2] + Lambda[4] +
                Lambda[8] + Lambda[14] + Lambda[13] +
                Lambda[11] + Lambda[7])*(1./8)
    def VmA(Lambda):
        return 2*(V(Lambda) - A(Lambda))/(V(Lambda) + A(Lambda))
    def PmS(Lambda):
        return 2*(Lambda[15] - Lambda[0])/(Lambda[15] + Lambda[0])
        
    Data.Lambda_VpA_JK = map(VpA, Data.LambdaJK)
    Data.Lambda_VpA_sigmaJK = JKsigma(Data.Lambda_VpA_JK, Data.Lambda_VpA)
    Data.Lambda_V_JK = map(V, Data.LambdaJK)
    Data.Lambda_V_sigmaJK = JKsigma(Data.Lambda_V_JK, Data.Lambda_V)
    Data.Lambda_A_JK = map(A, Data.LambdaJK)
    Data.Lambda_A_sigmaJK = JKsigma(Data.Lambda_A_JK, Data.Lambda_A)
    Data.Lambda_V_sigmaJK = JKsigma(Data.Lambda_V_JK, Data.Lambda_V)
    Data.Lambda_VmA_JK = map(VmA, Data.LambdaJK)
    Data.Lambda_VmA_sigmaJK = JKsigma(Data.Lambda_VmA_JK, Data.Lambda_VmA)
    Data.Lambda_PmS_JK = map(PmS, Data.LambdaJK)
    Data.Lambda_PmS_sigmaJK = JKsigma(Data.Lambda_PmS_JK, Data.Lambda_PmS)
    
    # For the (X, q) schemes.
    aq = Data.aq
    apSq = Data.apSq
    def Vq(amputated):
        Gmu = (amputated[1], amputated[2], amputated[4], amputated[8])
        qmuGmu = sum([aq[i]*Gmu[i] for i in range(4)])
        return (trace(dot(qmuGmu, dw.slash(aq))).real)*norm/apSq
    def Vq5(amputated):
        Gmu5 = (amputated[14], -amputated[13], amputated[11], -amputated[7])
        qmuGmu5 = sum([aq[i]*Gmu5[i] for i in range(4)])
        tmp = reduce(dot, [qmuGmu5, Gc[15], dw.slash(aq)]).real
        return trace(tmp*norm)/Data.apSq
    Data.Vq_JK = [(Vq(amp)+Vq5(amp))/2 for amp in amputatedJK]
    Data.Vq_sigmaJK = JKsigma(Data.Vq_JK, Data.Vq)
    
    # Z-factors.
    def Z_S(Lambda, Lambda_vec):
        'Lambda_vec depends on how Zq is determined (see usage below).'
        return Lambda_vec/Lambda[0]
        
    Data.Z_S_JK = dict(g=None, q=None)
    Data.Z_S_sigmaJK = dict(g=None, q=None)
    Data.Z_S_JK['g'] = map(Z_S, Data.LambdaJK, Data.Lambda_VpA_JK)
    Data.Z_S_JK['q'] = map(Z_S, Data.LambdaJK, Data.Vq_JK)
    for scheme in 'g', 'q':
        Data.Z_S_sigmaJK[scheme] = JKsigma(Data.Z_S_JK[scheme], Data.Z_S[scheme])
    
    
def bilinear_LambdaBoot(Data):
    pass

# Fourquark Stuff.
def amp_KmK(inprop, outprop, fourquark):
    iprop_inT = np.transpose(inv(inprop))
    iprop_out = inv(outprop)
    KmK = fourquark - np.transpose(fourquark, (0, 3, 2, 1))  # i j k l
    tmp1 = tensordot(iprop_out, KmK, [1, 0])   # i' j  k  l
    tmp2 = tensordot(iprop_inT, tmp1, [1, 1])  # j' i' k  l
    tmp3 = tensordot(iprop_out, tmp2, [1, 2])  # k' j' i' l
    tmp4 = tensordot(iprop_inT, tmp3, [1, 3])  # l' k' j' i'
    return np.transpose(tmp4)                  # i' j' k' l'

def amputate_fourquark(inprop_list, outprop_list, fourquark_array):
    "Amputate fourquark Green's functions."
    N = len(inprop_list)
    assert N == len(outprop_list) == len(fourquark_array)
    inprop = sum(inprop_list)/N
    outprop = sum(outprop_list)/N
    fourquark = sum(fourquark_array)/N
    return [amp_KmK(inprop, outprop, fourquark[g]) for g in range(16)]

def G_VVpAA(amputated):
    return amputated[1] + amputated[2] + amputated[4] + amputated[8] +\
           amputated[14] + amputated[13] + amputated[11] + amputated[7]

def G_VVmAA(amputated):
    return amputated[1] + amputated[2] + amputated[4] + amputated[8] -\
           amputated[14] - amputated[13] - amputated[11] - amputated[7]

def G_SSmPP(amputated):
    return amputated[0] - amputated[15]

def G_SSpPP(amputated):
    return amputated[0] + amputated[15]

def G_TT(amputated):
    return amputated[3] + amputated[5] + amputated[6] +\
           amputated[12] + amputated[10] + amputated[9]

def proj_g(amputated, gspec):
    "Contract amputated fourquark correlator w/ gamma*gamma projectors."
    result = []
    for g in gspec:
        sign = sgn(g)
        g = abs(g)
        tmp = sign*tensordot(Gc[g], amputated, ([0,1], [1,0]))
        result.append(tensordot(Gc[g], tmp, ([0,1], [1,0])))
    return sum(result)

# consolidate proj_q, proj_q5 w/ pseudo tag; q, qmix w/ color tag; g, q w/ ...
# split off a different file called projectors...
def proj_q(amputated, aq):
    "Contract amputated fourquark correlator w/ qslash x qslash projector."
    qslash = dw.slash(aq)
    tmp = tensordot(qslash, amputated, ([0,1], [1,0]))
    return tensordot(qslash, tmp, ([0,1], [1,0]))

def proj_q5(amputated, aq):
    "Contract amputated fourquark correlator w/ qslash5 x qslash5 projector."
    qslash5 = dot(dw.slash(aq), Gc[15])
    tmp = tensordot(qslash5, amputated, ([0,1], [1,0]))
    return tensordot(qslash5, tmp, ([0,1], [1,0]))

def proj_qmix(amputated, aq):
    proj = dw.qqMixArray(aq, False)
    #return tensordot(proj, amputated, ([1,0,3,2], [0,1,2,3]))
    return np.sum(proj*amputated)

def proj_q5mix(amputated, aq):
    proj = dw.qqMixArray(aq, True)
    #return tensordot(proj, amputated, ([1,0,3,2], [0,1,2,3]))
    return np.sum(proj*amputated)

def proj_sigma(amputated, aq):
    "Contract amputated fourquark correlator w/ sigma.q x sigma.q projector."
    result = 0
    sdq = dw.sigma_dot_q(aq, color=True)
    for mu in range(4):
        tmp = tensordot(sdq[mu], amputated, ([0,1], [1,0]))
        result += tensordot(sdq[mu], tmp, ([0,1], [1,0]))
    return result

def proj_sigma_mix(amputated, aq):
    "Contract amputated fourquark correlator w/ sigma.q x sigma.q mixed."
    proj = dw.sigmaMixArray(aq)
    return np.sum(proj*amputated)

def fourquark_proj_g(amputated):
    "Convert amputated Green functions into 5x5 matrix."
    P_VVpAA = [1, 2, 4, 8, 14, 13, 11, 7]
    P_VVmAA = [1, 2, 4, 8, -14, -13, -11, -7]
    P_SSmPP = [0, -15]
    P_SSpPP = [0, 15]
    P_TT = [3, 5, 6, 12, 10, 9]
    def projectors(x):
	    return [proj_g(x, P_VVpAA), proj_g(x, P_VVmAA), proj_g(x, P_SSmPP),
                proj_g(x, P_SSpPP), proj_g(x, P_TT)]
    cfncs = [G_VVpAA(amputated), G_VVmAA(amputated), G_SSmPP(amputated),
             G_SSpPP(amputated), G_TT(amputated)]

    return array(map(projectors, cfncs))

def fourquark_proj_q(amputated, aq, apSq):
    "Apply qslash projectors to amputated Green functions."
    proj_BK = lambda amp: (proj_q(amp, aq) + proj_q5(amp, aq))/apSq
    proj_VVmAA = lambda amp: (proj_q(amp, aq) - proj_q5(amp, aq))/apSq
    proj_VVmAA_mx = lambda amp: (proj_qmix(amp, aq) -
                                 proj_q5mix(amp, aq))/apSq  # SS-PP
    proj_TT_mx = lambda amp: proj_sigma_mix(amp, aq)/apSq   # SS+PP
    proj_TT = lambda amp: proj_sigma(amp, aq)/apSq
 
    def projectors(x):
        return [proj_BK(x), proj_VVmAA(x), proj_VVmAA_mx(x),
                proj_TT_mx(x), proj_TT(x)]

    cfncs = [G_VVpAA(amputated), G_VVmAA(amputated), G_SSmPP(amputated),
             G_SSpPP(amputated), G_TT(amputated)]

    return array(map(projectors, cfncs))
    
def fourquark_Zs(Data):
    "Z-factors corresponding to fourquark operators in (X,Y)-schemes."

    amputated = amputate_fourquark(Data.inprop_list,
				                   Data.outprop_list,
                                   Data.fourquark_array)
    norm = (Data.V)**3
    Data.fourquark_Zs = dict(gg=None, gq=None, qg=None, qq=None)
    # (g, Y) - schemes, deltaS = 2 basis
    Data.fourquark_Lambda = (fourquark_proj_g(amputated).real)*norm#*chiral_mask
    VpA = Data.Lambda_VpA  # Requires prior bilinear calculation.
    Vq = Data.Vq  # Requires prior bilinear calculation.
    Data.Zinv = dot(Data.fourquark_Lambda, inv(F_gg))
    Data.Zinv_chi = dot(Data.fourquark_Lambda*chiral_mask, inv(F_gg))
    Data.Z_chi = inv(Data.Zinv_chi)
    Data.thing = dot(Data.Z_chi, Data.Zinv)
    Data.Z_tmp = inv(Data.Zinv*chiral_mask)
    Data.fourquark_Zs['gg'] = Data.Z_tmp*(VpA)*(VpA)  # (g, g)
    Data.fourquark_Zs['gq'] = Data.Z_tmp*(Vq)*(Vq)  # (g, q)
    
    # (q, Y) - schemes, deltaS = 2 basis
    aq, apSq = Data.aq, Data.apSq
    Data.fourquark_Lambda_q = (fourquark_proj_q(amputated, aq, apSq).real)*\
                                                norm#*chiral_mask
    #Z_tmp = dot(F_qq, inv(Data.fourquark_Lambda_q))
    Data.Zinv_q = dot(Data.fourquark_Lambda_q, inv(F_qq))
    Data.Zinv_chi_q = dot(Data.fourquark_Lambda_q*chiral_mask, inv(F_qq))
    Data.Z_chi_q = inv(Data.Zinv_chi_q)
    Data.thing_q = dot(Data.Z_chi_q, Data.Zinv_q)
    Data.Z_tmpq = inv(Data.Zinv_q)
    Data.fourquark_Zs['qg'] = Data.Z_tmpq*(VpA)*(VpA) # (q, g)
    Data.fourquark_Zs['qq'] = Data.Z_tmpq*(Vq)*(Vq)    # (q, q)

def fourquark_ZsJK(Data):
    amputatedJK = map(amputate_fourquark, JKsample(Data.inprop_list),
                                          JKsample(Data.outprop_list),
                                          JKsample(Data.fourquark_array))
    norm = (Data.V)**3
    # Lambdas.
    Lambda = lambda amp: (fourquark_proj_g(amp).real)*norm#*chiral_mask
    Data.Lambda_JK = map(Lambda, amputatedJK)
    Data.Lambda_sigmaJK = JKsigma(Data.Lambda_JK, Data.fourquark_Lambda)
    
    # Zinv.
    Zinv = lambda Lambda, VpA: dot(Lambda, inv(F_gg))/(VpA*VpA)
    Data.Zinv_JK = map(Zinv, Data.Lambda_JK, Data.Lambda_VpA_JK)
    Data.Zinv_sigmaJK = JKsigma(Data.Zinv_JK, Data.Zinv)
    
    # Z^chi Lambda^f = "thing".
    thing = lambda x: reduce(dot, [F_gg, inv(Lambda(x)*chiral_mask),
                                   Lambda(x), inv(F_gg)])
    Data.thing_JK = map(thing, amputatedJK)
    Data.thing_sigmaJK = JKsigma(Data.thing_JK, Data.thing)
    
    # Lambda_q.
    aq, apSq = Data.aq, Data.apSq
    Lambda_q = lambda amp: (fourquark_proj_q(amp, aq, apSq).real)*\
                                                        norm#*chiral_mask
    Data.Lambda_JK_q = map(Lambda_q, amputatedJK)
    Data.Lambda_sigmaJK_q = JKsigma(Data.Lambda_JK_q, Data.fourquark_Lambda_q)
    
    # Z^chi Lambda_q^f = "thing_q".
    thing_q = lambda x: reduce(dot, [F_qq, inv(Lambda_q(x)*chiral_mask),
                                     Lambda_q(x), inv(F_qq)])
    Data.thing_q_JK = map(thing_q, amputatedJK)
    Data.thing_q_sigmaJK = JKsigma(Data.thing_q_JK, Data.thing_q)

    # Zs.
    Data.fourquark_ZsJK = dict(gg=None, gq=None, qg=None, qq=None)
    Data.fourquark_sigmaJK = dict(gg=None, gq=None, qg=None, qq=None)
    # (g, g)
    Zs = lambda Lambda, VpA: dot(F_gg, inv(Lambda))*VpA*VpA
    Data.fourquark_ZsJK['gg'] = map(Zs, Data.Lambda_JK, Data.Lambda_VpA_JK)
    # (g, q) 
    Zs = lambda Lambda, Vq: dot(F_gg, inv(Lambda))*Vq*Vq
    Data.fourquark_ZsJK['gq'] = map(Zs, Data.Lambda_JK, Data.Vq_JK)
    # (q, g)
    Zs = lambda Lambda, VpA: dot(F_qq, inv(Lambda))*VpA*VpA
    Data.fourquark_ZsJK['qg'] = map(Zs, Data.Lambda_JK_q, Data.Lambda_VpA_JK)
    # (q, q) 
    Zs = lambda Lambda, Vq: dot(F_qq, inv(Lambda))*Vq*Vq
    Data.fourquark_ZsJK['qq'] = map(Zs, Data.Lambda_JK_q, Data.Vq_JK)

    for scheme in 'gg', 'gq', 'qg', 'qq':
        Data.fourquark_sigmaJK[scheme] = JKsigma(Data.fourquark_ZsJK[scheme],
                                                 Data.fourquark_Zs[scheme])
    

def fourquark_ZsBoot(Data):
    pass

F_gg = array([[1536., 0, 0, 0, 0,],
              [0, 1152., -192., 0, 0],
              [0, -192., 288., 0, 0],
              [0, 0, 0, 240., 144.],
              [0, 0, 0, 144., 1008.]])

F_qq = array([[384, 0, 0, 0, 0],
              [0, 288, 96, 0, 0],
              [0, -48, -144, 0, 0],
              [0, 0, 0, 216, 72],
              [0, 0, 0, 360, 504]])

# Decouple the sectors.
chiral_mask = array([[1., 0, 0, 0, 0],
                     [0, 1., 1., 0, 0],
                     [0, 1., 1., 0, 0],
                     [0, 0, 0, 1., 1.],
                     [0, 0, 0, 1., 1.]])

three_mask = array([[1., 1., 1., 0, 0],
                    [1., 1., 1., 0, 0],
                    [1., 1., 1., 0, 0],
                    [0, 0, 0, 1., 1.],
                    [0, 0, 0, 1., 1.]])

