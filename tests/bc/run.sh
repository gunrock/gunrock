#!/bin/sh

#put OS and Device type here
SUFFIX="ubuntu12.04.k40c"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    echo ./bin/test_bc_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree
         ./bin/test_bc_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree --device=2 > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 1
done

for i in chesapeake test_bc
do
    echo ./bin/test_bc_6.0_x86_64 market ../../dataset/small/$i.mtx --src=-1
         ./bin/test_bc_6.0_x86_64 market ../../dataset/small/$i.mtx --src=-1 --device=2 > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 1
done
