#!/bin/sh

#put OS and Device type here
SUFFIX="linuxmin15.gtx680"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 webbase-1M
do
    echo ./bin/test_cc_5.5_x86_64 market ../../dataset/large/$i/$i.mtx
         ./bin/test_cc_5.5_x86_64 market ../../dataset/large/$i/$i.mtx > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 10
done


