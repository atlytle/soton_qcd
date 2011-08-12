import numpy as np
import pyNPR as npr
from domain_wall import inner, ap, mu

#np.set_printoptions(precision=4, suppress=True)

process = lambda x: '{0: .5f}'.format(x)

def write_Zs_gnu(data, location):
    '''Write fourquark Zs in simple format to be used by, e.g., gnuplot'''
    def line_out(d):
        dada = [d.apSq] + d.fourquark_Zs.real.reshape(25).tolist()\
                        + d.fourquark_sigmaJK.reshape(25).tolist()
        tmp = map(process, dada)
        return ' '.join(tmp)

    with open(location, 'w') as f:
        for d in data:
            f.write(line_out(d) + '\n')

def write_stepscale_gnu(data, location):
    '''Write step-scaling result in a format usable by gnuplot.'''
    def line_out(d):
        dada = [mu(d.ap, d.a)] +\
               (d.step_scale*npr.chiral_mask).reshape(25).tolist()
        tmp = map(process, dada)
        return ' '.join(tmp)

    with open(location, 'w') as f:
        for d in data:
            f.write(line_out(d) + '\n')


def write_Zs_TeX(data, location):
    '''Write fourquark Zs in a human readable output format, e.g. for TeX.'''
    def results_matrix(a55, a55error):
        return [['{0} +/-{1}'.format(process(a55[i][j]),
                                     process(a55error[i][j])) 
                                      for j in range(5)]
                                      for i in range(5)]

    def results_matrix_33(a33, a33error):
        return [['{0} +/-{1}'.format(process(a33[i][j]),
                                     process(a33error[i][j])) 
                                      for j in range(3)]
                                      for i in range(3)]
    def matrix_string(m):
        return '\n'.join(['    '.join(row) for row in m])
    
    with open(location, 'w') as f:
        # (g, g) - scheme
        f.write('(g, g) - scheme\n')
        for d in data:
            f.write('am = {0}\nap = {1}, tw = {2}\n--------------\n'.format(d.m, d.p, d.tw))
            f.write(matrix_string(results_matrix(d.fourquark_Zs,
                                                 d.fourquark_sigmaJK)))
            f.write('\n\n')
        # (g, q) - scheme
        f.write('(g, q) - scheme\n')
        for d in data:
            f.write('am = {0}\nap = {1}, tw = {2}\n--------------\n'.format(d.m, d.p, d.tw))
            f.write(matrix_string(results_matrix(d.fourquark_Zs_q,
                                                 d.fourquark_sigmaJK_q)))
            f.write('\n\n')
        # (q, g) - scheme
        f.write('(q, g) - scheme\n')
        for d in data:
            f.write('am = {0}\nap = {1}, tw = {2}\n--------------\n'.format(d.m, d.p, d.tw))
            f.write(matrix_string(results_matrix_33(d.fourquark_Zs_qg,
                                                    d.fourquark_sigmaJK_qg)))
            f.write('\n\n')
        # (q, q) - scheme
        f.write('(q, q) - scheme\n')
        for d in data:
            f.write('am = {0}\nap = {1}, tw = {2}\n--------------\n'.format(d.m, d.p, d.tw))
            f.write(matrix_string(results_matrix_33(d.fourquark_Zs_qq,
                                                    d.fourquark_sigmaJK_qq)))
            f.write('\n\n')

def write_Zs_TeX_2(data, location):
    '''
    Write fourquark Zs in a human readable output format, e.g. for TeX.
    
    Does the conversion to the 3x3 Delta S = 1 basis.
    '''

    def results_matrix(a33, a33error):
        return [['{0} +/-{1}'.format(process(a33[i][j]),
                                     process(a33error[i][j])) 
                                      for j in range(3)]
                                      for i in range(3)]
    def matrix_string(m):
        return '\n'.join(['    '.join(row) for row in m])

    def new(Zs):
        # Delta S = 2 --> Delta S = 1.
        convert = np.array([[1, 1, -0.5],
                            [1, 1, -0.5],
                            [-2, -2, 1]])
        return convert*Zs[:3,:3]
     
    def new2(Zs):
        # Scale error bars.
        convert = np.array([[1, 1, 0.5],
                            [1, 1, 0.5],
                            [2, 2, 1]])
        return convert*Zs[:3,:3]
    
    with open(location, 'w') as f:
        # (g, g) - scheme
        f.write('(g, g) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix(new(d.fourquark_Zs),
                                                 new2(d.fourquark_sigmaJK))))
            f.write('\n\n')
        # (g, q) - scheme
        f.write('(g, q) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix(new(d.fourquark_Zs_q),
                                                 new2(d.fourquark_sigmaJK_q))))
            f.write('\n\n')
        # (q, g) - scheme
        f.write('(q, g) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix(new(d.fourquark_Zs_qg),
                                                 new2(d.fourquark_sigmaJK_qg))))
            f.write('\n\n')
        # (q, q) - scheme
        f.write('(q, q) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix(new(d.fourquark_Zs_qq),
                                                 new2(d.fourquark_sigmaJK_qq))))
            f.write('\n\n')

def write_stepscale_TeX(data, location):
    '''Write step-scaling functions in a human readable output format.'''

    def results_matrix(a55, a55error):
        return [['{0} +/-{1}'.format(process(a55[i][j]),
                                     process(a55error[i][j])) 
                                      for j in range(5)]
                                      for i in range(5)]

    def results_matrix_33(a33, a33error):
        return [['{0} +/-{1}'.format(process(a33[i][j]),
                                     process(a33error[i][j])) 
                                      for j in range(3)]
                                      for i in range(3)]
    def matrix_string(m):
        return '\n'.join(['    '.join(row) for row in m])
    
    with open(location, 'w') as f:
        # (g, g) - scheme
        f.write('(g, g) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix(d.step_scale,
                                                 d.step_scale_sigma)))
            f.write('\n\n')
        # (g, q) - scheme
        f.write('(g, q) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix(d.step_scale_q,
                                                 d.step_scale_sigma_q)))
            f.write('\n\n')
        # (q, g) - scheme
        f.write('(q, g) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix_33(d.step_scale_qg,
                                                    d.step_scale_sigma_qg)))
            f.write('\n\n')
        # (q, q) - scheme
        f.write('(q, q) - scheme\n')
        for d in data:
            f.write('am = {0}\nmu = {1:.4f}\n--------------\n'.format(d.m, mu(d.ap, d.a)))
            f.write(matrix_string(results_matrix_33(d.step_scale_qq,
                                                    d.step_scale_sigma_qq)))
            f.write('\n\n')


def write_Zs_mma(data, location):
    '''Write Z matrices as Mathematica expressions.'''
    pass

    def to_matrix(m):  # Numpy array to MMA matrix.
        result = ['{']
        for i in range(5):
            result.append('{')
            for j in range(4):
                result.append(process(m[i][j]) + ',')
            result.append(process(m[i][4]) + '}')
            result.append(', ')
        result = result[:-1]  # Remove last comma.
        result.append('}')
        return ''.join(result)

    def to_list(p):  # Form MMA list from python tuple.
        s = ','.join(map(str, p))
        return '{' + s + '}'

    def mma_defs(d):  # Construct MMA assignments.
        s = 'fourquarkZ[{0},{1},{2}] = {3};\n'\
            'fourquarkZJK[{0},{1},{2}] = {4};\n'.format(d.m, to_list(d.p), d.tw,
                                                    to_matrix(d.fourquark_Zs),
                                                    to_matrix(d.fourquark_sigmaJK))
        return s
        
    with open(location, 'w') as f:
        for d in data:
            f.write(mma_defs(d))

def write_stepscale_mma(data, location):
    '''Write step-scaling functions as Mathematica expressions.'''
    pass

    def to_matrix(m):  # Numpy array to MMA matrix.
        result = ['{']
        for i in range(5):
            result.append('{')
            for j in range(4):
                result.append(process(m[i][j]) + ',')
            result.append(process(m[i][4]) + '}')
            result.append(', ')
        result = result[:-1]  # Remove last comma.
        result.append('}')
        return ''.join(result)

    def to_list(p):  # Form MMA list from python tuple.
        s = ','.join(map(str, p))
        return '{' + s + '}'

    def mma_defs(d):  # Construct MMA assignments.
        s = 'ss[{0}, {1}, {2}] = {{{3}, {4}}};\n'\
            'ssJK[{0}, {1}, {2}] = {{{3}, {5}}};\n'.format(
                                            d.m, to_list(d.p),
                                            d.tw, mu(d.ap, d.a),
                                            to_matrix(d.step_scale),
                                            to_matrix(d.step_scale_sigma))
        return s
        
    with open(location, 'w') as f:
        for d in data:
            f.write(mma_defs(d))


def print_step_scaling(data):
    def line_out(d):
        foo = mu(d.apSq) + d.step_scale.real.reshape(25).tolist()
        bar = map(process, foo)
        return ' '.join(bar)

        for d in data:
            print line_out(d)
