#!/usr/bin/env python

# Find Top K degree centrality node_ids and URLs
# Input: gpu_result_file mapping_info_file index_file
# Output format: | node_id | #degrees | URL |

import sys
import fileinput

# check input format
if len(sys.argv) != 4:
    print 'Usage: python top_k.py out_file map_info index_file'
    sys.exit()

# put mapping information into a dictionary
map_dict = {}
# for item in map_dict
for line in fileinput.input(sys.argv[2]):
    (key, val) = line.split()
    map_dict[int(key)] = val

# put index information into a dictionary
index_dict = {}
for line in fileinput.input(sys.argv[3]):
    (val, key) = line.split()
    index_dict[int(key)] = val

# find original vertex ids and URLs
out_file = open(sys.argv[1])
results = open('results.txt', 'w')
for line0 in out_file.readlines():
    (vertex_id, num_degrees) = line0.split()
    #print "Results  Vertex ID:" + str(vertex_id) + '\t\t',
    #print "Original Vertex ID:" + str(map_dict[int(vertex_id)])
    for line1 in fileinput.input(sys.argv[3]):
        (URL, original_id) = line1.split()
        if original_id == vertex_id:
            print str(original_id) + '\t' + str(num_degrees) + '\t' + str(index_dict[int(original_id)])
            # write line back to the output file
            new_line = str(original_id) + '\t' + str(num_degrees) + '\t' + str(index_dict[int(original_id)]) + '\n'
            results.write(new_line)
results.close()
# end
