import sys
from sets import Set
from math import sqrt

edge_file = open('cities.mtx', 'w')
map_file  = open('cities.map', 'w')

if len(sys.argv) != 3:
    print "Usage: python gen_graph.py filename threshold"
else:
    #get (vid, name, lat, lon, loc) tuple
    script, fname, threshold = sys.argv
    threshold = float(threshold)
    with open(fname) as f:
        lines = f.readlines()

    idx = 0
    vids = []
    names = []
    lat = []
    lon = []
    loc = []
    for line in lines:
        if line[0] == 'I' or line[0] == 'z':
            continue
        items = line.split()
        vids.append(idx)
        names.append(items[0])
        lat.append(float(items[1]))
        lon.append(float(items[2]))
        loc.append(''.join(items[3:]))
        idx += 1

    res = ''
    num_edges = 0
    nodes = Set()
    for i in range(len(vids)-1):
        for j in range(i+1,len(vids)):
            distance = (lat[i]-lat[j])**2+(lon[i]-lon[j])**2
            if distance <= threshold**2:
                res += str(i+1) + ' ' + str(j+1) + ' ' + str(sqrt(distance)) + '\n'
                num_edges += 1
                nodes.add(i)
                nodes.add(j)
    num_nodes = len(nodes)
    head = str(num_nodes) + ' ' + str(num_nodes) + ' ' + str(num_edges) + '\n'
    edge_file.write(head)
    edge_file.write(res)
    edge_file.close()

    for each in nodes:
        line = str(each) + ' ' + names[each] + ' ' + str(lat[each]) + ' ' + str(lon[each]) + '\n'
        map_file.write(line)
    map_file.close()

        

