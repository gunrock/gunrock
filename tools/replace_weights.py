#!usr/bin/python

"""
simple python script to generate random weight value for .mtx graph
used for replacing original associated weight such as all 1s.
"""

import random
import fileinput
import sys

### check command line args
if len(sys.argv) != 2:
    print 'Usage: python replace_weights.py file_name.mtx'
    sys.exit()

### output matrix-market graph
ograph = open(sys.argv[1].split('.')[0] + '.random.weight.mtx', 'w')
is_edge = False
for line in fileinput.input(sys.argv[1]):
    if line[0] == '%' or line == '\n':
        ograph.write(line)
    elif is_edge == False:
        is_edge = True
        ograph.write(line)
    elif is_edge == True:
        line = line.split(' ')
        res = line[0] + ' ' + line[1] + ' ' + str(random.randint(1, 63)) + '\n'
        ograph.write(res)
ograph.close()
