# python script to generate mtx graph

import fileinput
import sys
from sets import Set

if (len(sys.argv)) != 2:
    print 'Usage: python read_file.py file_name'
    sys.exit()

edge_num = -1 # ignore the table header
nodes = Set()
dict = {}
node_num = 0

for line in fileinput.input(sys.argv[1]):
    edge_num += 1
    if edge_num > 0:
        start = line.split()
        nodes.add(start[0])
        if (dict.get(start[0], -1) is -1):
            dict[start[0]] = node_num
            node_num += 1
        nodes.add(start[1])
        if (dict.get(start[1], -1) is -1):
            dict[start[1]] = node_num
            node_num += 1

line_num = 1

for line in fileinput.input(sys.argv[1], inplace = 1):
    start = line.split()
    if line_num == 1:
        line_num += 1
        line = ''
        line += str(len(nodes))
        line += ' '
        line += str(len(nodes))
        line += ' '
        line += str(edge_num+1)
        line += '\n'
        print line,
    if line_num > 1:
        line = ''
        line += str(dict[start[1]]+1)
        line += ' '
        line += str(dict[start[0]]+1)
        line += '\n'
        print line,

# end
