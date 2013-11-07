#!/bin/sh

#put OS and Device type here
SUFFIX="linuxmint15.gtx680"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13
do
    echo ./bin/test_bc_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=-1
         ./bin/test_bc_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=-1 > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 10
done

for i in delaunay_n21 webbase-1M
do
    echo ./bin/test_bc_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --quick --src=-1
         ./bin/test_bc_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --quick--src=-1 > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 10
done
