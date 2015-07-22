#!/usr/bin/env python

import os

for mark_pred in ["", "--mark-pred"]:
    for directed in ["", "--undirected"]:
        for dataset in ['soc-LiveJournal1',
                        'kron_g500-logn21',
                        'delaunay_n21',
                        'belgium_osm',
                        'ak2010',
                        'delaunay_n13',
                        'coAuthorsDBLP']:
            os.system("../../gunrock-build/bin/breadth_first_search market ../dataset/large/%s/%s.mtx --src=0 %s %s --idempotence --iteration-num=10 --quiet --jsondir=." % (dataset, dataset, mark_pred, directed))
