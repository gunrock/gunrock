#!/usr/bin/env python

import os

datasets = ['soc-LiveJournal1',
            'kron_g500-logn21',
            'delaunay_n21',
            'delaunay_n24',
            'belgium_osm',
            'europe_osm',
            'road_usa',
            'cit-Patents',
            'ak2010',
            'delaunay_n13',
            'coAuthorsDBLP']

options = {
    "direction_optimizing_bfs" : "--src=0 --idempotence",
    "breadth_first_search" : "--src=0 --idempotence",
    "betweenness_centrality": "--src=0",
    "connected_component": "",
    "pagerank": "--undirected",
    "single_source_shortest_path": "--src=0 --undirected",
}


for binary in ["direction_optimizing_bfs", "breadth_first_search"]:
    for mark_pred in ["", "--mark-pred"]:
        for directed in ["", "--undirected"]:
            for dataset in datasets:
                os.system("../../gunrock-build/bin/%s market ../dataset/large/%s/%s.mtx %s %s %s --iteration-num=10 --quiet --jsondir=." % (binary, dataset, dataset, options[binary], mark_pred, directed))

for binary in ["betweenness_centrality",
               "connected_component",
               "pagerank",
               "single_source_shortest_path"]:
    for dataset in datasets:
        os.system("../../gunrock-build/bin/%s market ../dataset/large/%s/%s.mtx %s --iteration-num=10 --quiet --jsondir=." % (binary, dataset, dataset, options[binary]))
