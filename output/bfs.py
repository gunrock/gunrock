#!/usr/bin/env python

import os

for binary in ["direction_optimizing_bfs", "breadth_first_search"]:
    for mark_pred in ["", "--mark-pred"]:
        for directed in ["", "--undirected"]:
            for dataset in ['soc-LiveJournal1',
                            'kron_g500-logn21',
                            'delaunay_n21',
                            'delaunay_n24',
                            'belgium_osm',
                            'europe_osm',
                            'road_usa',
                            'cit-Patents',
                            'ak2010',
                            'delaunay_n13',
                            'coAuthorsDBLP']:
                os.system("../../gunrock-build/bin/%s market ../dataset/large/%s/%s.mtx --src=0 %s %s --idempotence --iteration-num=10 --quiet --jsondir=." % (binary, dataset, dataset, mark_pred, directed))
