#!usr/bin/python

"""
simple python script to remove weight values associated with .mtx graph
"""

import random
import fileinput
import sys

### check command line args
if len(sys.argv) != 2:
    print 'Usage: python remove_weights.py file_name.mtx'
    sys.exit()

### output matrix-market format graph
ograph = open(sys.argv[1].split('.')[0] + '.no.weight.mtx', 'w')
is_edge = False
for line in fileinput.input(sys.argv[1]):
    if line[0] == '%' or line == '\n':
        ograph.write(line)
    elif is_edge == False:
        is_edge = True
        ograph.write(line)
    elif is_edge == True:
        line = line.split(' ')
        new_line = line[0] + ' ' + line[1] + '\n'
        ograph.write(new_line)
ograph.close()
