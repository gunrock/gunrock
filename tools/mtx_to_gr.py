#!/usr/local/bin/python

"""
Simple script to convert matrix-market formatted graph to .gr graph
    gr file: [http://www.dis.uniroma1.it/challenge9/format.shtml#graph]
    example: [http://www.dis.uniroma1.it/challenge9/samples/sample.gr]
"""

import os
import sys
import string

if (len(sys.argv)) != 2:
    print ' Usage: python mtx_to_gr.py file_name.mtx'
    sys.exit()

### matrix-market mtx
file_mm = sys.argv[1]

### GTGraph (.gr) format output file
file_gr = sys.argv[1].split('.')[0] + ".gr"

line_num = 0;
with open(file_mm, 'r') as mm, open(file_gr, 'w') as gr:
    for line in mm:
        ### skip blank lines and comments
        if line.strip() == '' or '%' in line:
            pass
        else:
            if line_num == 0:
                ### write first line -> p sp nodes edges
                item = line.split(' ')
                n = item[0]
                e = item[2].split()[0]
                write = 'p ' + 'sp ' + str(n) + ' ' + str(e) + '\n'
                gr.write(write)
                line_num = 1
            else:
                ### write rest of mtx contents -> a u v w
                item = line.split(' ')
                v = item[0]
                u = item[1].split()
                u = u[0]
                if len(item) == 2:  ### add weight 1 if absent
                    write = 'a ' + str(u) + ' ' + str(v) + ' 1'+ '\n'
                elif len(item) == 3:  ### write weight if exist
                    w = item[2].split()[0]
                    write = 'a ' + str(u) + ' ' + str(v) + ' ' + str(w) + '\n'
                gr.write(write)
mm.close()
gr.close()
