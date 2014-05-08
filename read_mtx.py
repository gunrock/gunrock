#!usr/bin/python
# python script to format the charity_net_graph data
# file: read_mtx.py

import sys
import fileinput
from sets import Set

if (len(sys.argv)) != 3:
    print 'Usage: python readfile.py intput output'
    sys.exit()

# calculate graph size
num_edges = 0
for line in fileinput.input(sys.argv[1]):
    num_edges += 1
    start = line.split()
    num_nodes = int(start[0]) + 1

# convert to mtx format
line_num = 1
for line in fileinput.input(sys.argv[1]):
    start = line.split()
    if line_num == 1:
        line_num += 1
        line = ''
        line += str(num_nodes)
        line += ' '
        line += str(num_nodes)
        line += ' '
        line += str(num_edges)
        line += '\n'
        # write back into mtx file
        with open(sys.argv[2], 'a') as f:
            f.write(line)
        f.close()
    if line_num > 1:
        line = ''
        line += str(int(start[1])+1)
        line += ' '
        line += str(int(start[0])+1)
        line += '\n'
        # write back into mtx file
        with open(sys.argv[2], 'a') as f:
            f.write(line)
        f.close()
         
