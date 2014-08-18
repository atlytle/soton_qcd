import numpy
from numpy import array

numpy.set_printoptions(suppress=True)
nico_old_gg = array([[.42074, 0., 0.],
                     [0., .47855, -.02448],
                     [0., -.04530, .54274]])
nico_new_gg = array([[.41850, 0., 0.],
                     [0., .47896, -.02226],
                     [0., -.04731, .55191]])
andy_old_gg = array([[.42061, 0., 0.],
                     [0., .47801, -.02552],
                     [0., -.04398, .54047]])
andy_new_gg = array([[.42023, 0., 0.],
                     [0., .47787, -.02319],
                     [0., -.0449, .55035]])

nico_old_qq = array([[.42699, 0., 0.],
                     [0., .47295, -.02596],
                     [0., -.07025, .56420]])
nico_new_qq = array([[.42442, 0., 0.],
                     [0., .47245, -.01970],
                     [0., -.06749, .57228]])
andy_old_qq = array([[.42679, 0., 0.],
                     [0., .47299, -.02553],
                     [0., -.06792, .5652]])
andy_new_qq = array([[.42603, 0., 0.],
                     [0., .47314, -.02364],
                     [0., -.06708, 0.5739]])

andy_newa_gg = array([[.42027, 0., 0.],
                      [0., .47825, -.02394],
                      [0., -.04524, .54922]])
andy_newa_qq = array([[.42639, 0., 0.],
                      [0., .47376, -.02449],
                      [0., -.06698, .5736]])

if __name__ == "__main__":
    print "Nicolas diff gg:"
    print nico_new_gg - nico_old_gg
    print "Andrew diff gg:"
    print andy_new_gg - andy_old_gg

    print ''
    print "Nicolas diff qq:"
    print nico_new_qq - nico_old_qq
    print "Andrew diff qq:"
    print andy_new_qq - andy_old_qq

    print ''
    print "Andrew delta_a effect gg:"
    print andy_newa_gg - andy_new_gg   
    print "Andrew delta_a effect qq:"
    print andy_newa_qq - andy_new_qq
