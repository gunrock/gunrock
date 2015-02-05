# python script to generate mtx graph

import fileinput
import sys
from sets import Set

if (len(sys.argv)) != 3:
    print 'Usage: python read_file.py input_file_name mapping_file_name'
    sys.exit()

edge_num = -1 # ignore the table header
nodes = Set()
dict = {}
node_num = 0

for line in fileinput.input(sys.argv[1]):
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

line_num = 1
mlist = []
for key,value in dict.iteritems():
    temp = [key, value]
    mlist.append(temp)
mlist.sort(key=lambda x: x[1])

final_out = []
for item in mlist:
    final_out.append(item[0] + ' ' + str(item[1]) + '\n')

f = open(sys.argv[2], 'w')
for each in final_out:
    f.write(each)
f.close()

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
        line += str(dict[str(start[1])]+1)
        line += ' '
        line += str(dict[str(start[0])]+1)
        line += '\n'
        print line,

# end
