#!/bin/sh

#put OS and Device type here
SUFFIX="ubuntu12.04.k40c"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    echo ./bin/test_topk_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --top=1000
         ./bin/test_topk_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --top=1000 > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 1
done