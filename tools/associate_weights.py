#!usr/bin/python

"""
Simple python script to generate random weight values for file_name.mtx graph
"""

import random
import fileinput
import sys

### check command line args
if len(sys.argv) != 2:
    print 'Usage: python associate_weights.py file_name.mtx'
    sys.exit()

### Associate random weight (0, 63) values for each edge in the graph
ograph = open(sys.argv[1].split('.')[0] + '.random.weight.mtx', 'w')
is_edge = False
for line in fileinput.input(sys.argv[1]):
    if line[0] == '%' or line == '\n':
        ograph.write(line)
    elif is_edge == False:
        is_edge = True
        ograph.write(line)
    elif is_edge == True:
        line = line.split('\n')
        new_line = line[0] + ' ' + str(random.randint(1, 63)) + '\n'
        ograph.write(new_line)

ograph.close()
