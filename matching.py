from numpy import array, pi, identity, zeros

# alpha_s from mma notebook.
alpha_s2 = 0.296008  # 2 GeV
alpha_s3 = 0.245442  # 3 GeV

xi = 0  # Gauge-fixing parameter.

# From B_K paper & Christoph's note.
R_BK = {'gg': 0.2118 + 0.734*xi,
        'gq': -2.4548 - 1.06*xi,
        'qg': 2.2118 + 2.08*xi,
        'qq': -0.4548 + 0.286*xi,}

# From Christoph's note.
R_78 = {'gg': array([[0.0432 - 0.2381*xi, -0.1296 + 0.7143*xi],
                     [-1.6137 + 0.2143*xi, 4.4956 + 1.2619*xi]]),

        'gq': array([[-2.6235 - 2.0300*xi, -0.1296 + 0.7143*xi],
                     [-1.6137 + 0.2143*xi, 1.8289 - 0.5300*xi]]),

        'qg': array([[2.7099 + 1.5538*xi, -0.1296 + 0.7143*xi],
                     [-0.6137 + 0.8863*xi, 4.1623 + 1.0379*xi]]),

        'qq': array([[0.0432 - 0.2381*xi, -0.1296 + 0.7143*xi],
                     [-0.6137 + 0.8863*xi, 1.4956 - 0.7540*xi]])}

def C_BK(alpha_s, scheme):
    "O_BK matching factor (SMOM scheme -> NDR)."
    return 1 + (alpha_s/(4*pi))*R_BK[scheme]

def C_78(alpha_s, scheme):
    "O_7, O_8 matching factor (SMOM scheme -> NDR)."
    return identity(2) + (alpha_s/(4*pi))*R_78[scheme]

def C_178(alpha_s, scheme):
    "O_BK, O_7, O8 matching factor (SMOM scheme -> NDR)."
    tmp = zeros((3,3))
    tmp[0,0] = C_BK(alpha_s, scheme)
    tmp[1:3,1:3] = C_78(alpha_s, scheme)
    return tmp
