#!/usr/bin/env python

# find top K degree centrality node_ids and URLs
# file: top_k.py
# input:       | gpu_output_results  | mapping_info_file | index_file |
# output text: | node_ids (original) | degree centrality |    URL     |
# output file: results.txt & results.json

import sys
import fileinput
import csv

# check input
if len(sys.argv) != 4:
    print 'Usage: python top_k.py <gpu_output> <map_info> <index>'
    sys.exit()

# put mapping information into a dictionary
map_dict = {}
for line in fileinput.input(sys.argv[2]):
    (key, val) = line.split()
    map_dict[int(key)] = val

# put index information into a dictionary
index_dict = {}
for line in fileinput.input(sys.argv[3]):
    (val, key) = line.split()
    index_dict[int(key)] = val

# find original vertex ids and URLs
out_file = open(sys.argv[1]) # open gpu_output text file
results = open('results.txt', 'w')

head = 'node degree_centrality url\n'
results.write(head)

for line in out_file.readlines():
    # skip blank lines
    if not line.strip(): 
        continue
    else:
        (node, dc) = line.split()
        #print "gpu results ndoe id:" + str(node) + '\t\t',
        #print "original node id:" + str(map_dict[int(node)])
        for key in index_dict:
            if key == int(node):
                #print str(key) + ' ' + str(dc) + ' ' + str(index_dict[int(key)])
                # write new line back into the output file: results.txt
                nl = str(key) + ' ' + str(dc) + ' ' + str(index_dict[int(key)]) + '\n'
                results.write(nl)
        
results.close()

# Covert to JSON format output
jsfile = file('results.json', 'w')
jsfile.write('[\r\n')

with open('results.txt', 'r') as f:
    next(f) # skip headings
    reader = csv.reader(f, delimiter = ' ')
    
    # get the total number of rows excluded the heading
    row_count = len(list(reader))
    ite = 0
    
    # back to the first position
    f.seek(0)
    next(f)
    
    # write results
    for node,dc,url in reader:
        ite += 1
        jsfile.write('\t{\r\n')
        
        n = '\t\t\"node_id\": ' + node + ',\r\n'
        d = '\t\t\"degree_centrality\": ' + dc + ',\r\n'
        u = '\t\t\"website_link\": \"' + url + '\"\r\n'
                                                      
        jsfile.write(n)
        jsfile.write(d)
        jsfile.write(u)

        jsfile.write('\t}')

        # omit comma for last row item
        if ite < row_count:
            jsfile.write(',\r\n')

        jsfile.write('\r\n')

jsfile.write(']')
jsfile.close()

print '==> Complete.'
print 'TEXT output: results.txt'
print 'JSON output: results.json'

# end
