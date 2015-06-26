#!/usr/bin/python

import os

for kernel in ['plain', 'blas', 'block4', 'block8', 'block16', 'block32', 'block64']:
    for k in range(8,12):
        cmd = './gemm ' + str(2**k) + ' 1 ' + kernel
        os.system(cmd)

