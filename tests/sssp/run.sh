#!/bin/sh

OPTION="--undirected" #undirected graph"

#put OS and Device type here
SUFFIX="ubuntu12.04.k40c"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    echo ./bin/test_sssp_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree
         ./bin/test_sssp_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree > eval/$SUFFIX/$i.$SUFFIX.dir.txt
    sleep 1
    echo ./bin/test_sssp_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION
         ./bin/test_sssp_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION > eval/$SUFFIX/$i.$SUFFIX.undirected.txt
    sleep 1
done
