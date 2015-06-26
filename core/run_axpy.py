#!/usr/bin/python

import os

for kernel in ['plain', 'blas']:
    for k in range(4,25):
        cmd = './axpy ' + str(2**k) + ' ' + str(2**(30-k)) + ' ' + kernel
        os.system(cmd)

