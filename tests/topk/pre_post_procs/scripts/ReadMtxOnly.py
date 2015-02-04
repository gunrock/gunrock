# python script to generate mtx graph
# generate mtx graph only without storing mapping information

import fileinput
import sys
from sets import Set

if (len(sys.argv)) != 2:
    print 'Usage: python read_file.py input_file_name mapping_file_name'
    sys.exit()

edge_num = -1 # ignore the table header
nodes = Set()
dict = {}
node_num = 0

# replace the original file
for line in fileinput.input(sys.argv[1], inplace = 1):
    edge_num += 1
    if edge_num >= 0:
        start = line.split()
        nodes.add(str(start[0]))
        if (dict.get(str(start[0]), '') is ''):
            dict[str(start[0])] = node_num
            node_num += 1
        nodes.add(str(start[1]))
        if (dict.get(str(start[1]), '') is ''):
            dict[str(start[1])] = node_num
            node_num += 1
    line = ''
    line += str(dict[str(start[1])]+1)
    line += ' '
    line += str(dict[str(start[0])]+1)
    line += '\n'
    print line,

# put the print head into mtx file
print str(node_num) + ' ' + str(node_num) + ' ' + str(edge_num+1)

# end
