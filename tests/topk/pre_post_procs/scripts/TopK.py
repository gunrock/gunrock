#!/usr/bin/env python

# find top K degree centrality node_ids and URLs
# file: top_k.py
# input:       | gpu_output_results  | mapping_info_file | index_file |
# output format:  node ID: ( url, degree centrality )
# output file: top_k_results.json

import sys
import fileinput
import json
from collections import OrderedDict

# check input
if len(sys.argv) != 4:
    print 'Usage: python top_k.py <gpu_output> <map_info> <index>'
    sys.exit()

# put mapping information into a dictionary
map_dict = {}
for line in fileinput.input(sys.argv[2]):
    (val, key) = line.split()
    map_dict[int(key)] = val

# put index information into a dictionary
idx_dict = {}
for line in fileinput.input(sys.argv[3]):
    (val, key) = line.split()
    idx_dict[int(key)] = val

# find original vertex ids and URLs
out_dict = OrderedDict()
for line in fileinput.input(sys.argv[1]):
    # skip blank lines
    if not line.strip():
        continue
    else:
        (node, dc) = line.split()
        #print "gpu results ndoe id:" + str(node) + '\t\t',
        #print "original node id:" + str(map_dict[int(node)])
        ori_node = map_dict[int(node)]
        url = idx_dict[int(ori_node)]
        out_dict[int(ori_node)] = url, int(dc)

# write results back to a json format file
js_results = json.dumps([{'centrality':val[1], 'id':key, 'name':val[0]} for key, val in out_dict.items()], indent=4)
results = open('top_k.json', 'w')
print >> results, js_results
results.close()

print '==> Complete.'
print 'JSON output: top_k.json'

# end
