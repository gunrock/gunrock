#!/usr/local/bin/python

"""
Simple python script to convert .gr format graph to .mtx format
"""

import os
import sys
import string

### check command line args
if (len(sys.argv)) != 2:
    print ' Usage: python gr_to_mtx_symmetric.py graph.gr'
    sys.exit()

### gr graph input
file_gr = sys.argv[1]

### matrix-market format output file
file_mm = sys.argv[1].split('.')[0] + ".symmetric.mtx"

line_num = 0;
with open(file_gr, 'r') as gr, open(file_mm, 'w') as mm:
    mm.write('%%MatrixMarket matrix coordinate Integer symmetric\n')
    for line in gr:
        ### skip blank lines and comments
        if line.strip() == '' or 'c' in line:
            pass
        else:
            item = line.split(' ')
            if item[0] == 'p':
                ### write first line -> nodes nodes edges
                n = item[2]
                e = item[3].split()
                e = e[0]
                write = str(n) + ' ' + str(n)+ ' ' + str(e) + '\n'
                mm.write(write)
            if item[0] == 'a':
                ### write rest of mtx contents -> dst src wight
                v = item[1]
                u = item[2]
                w = item[3].split()
                w = w[0]
                write = str(u) + ' ' + str(v) + ' ' + str(w) + '\n'
                mm.write(write)
gr.close()
mm.close()
