#!/bin/sh

OPTION1="" #directed and do not mark-pred"
OPTION2="--mark-pred" #directed and mark-pred"
OPTION3="--undirected" #undirected and do not mark-pred"
OPTION4="--undirected --mark-pred" #undirected and mark-pred"

#put OS and Device type here
SUFFIX="linuxmint15.gtx680"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 kron_g500-logn21 webbase-1M
do
    echo ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION1
         ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION1 > eval/$SUFFIX/$i.$SUFFIX.dir_no_mark_pred.txt
    sleep 10
    echo ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION2
         ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION2 > eval/$SUFFIX/$i.$SUFFIX.dir_mark_pred.txt
    sleep 10
    echo ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION3
         ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION3 > eval/$SUFFIX/$i.$SUFFIX.undir_no_mark_pred.txt
    sleep 10
    echo ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION4
         ./bin/test_dobfs_5.5_x86_64 market ../../dataset/large/$i/$i.mtx --src=randomize $OPTION4 > eval/$SUFFIX/$i.$SUFFIX.undir_mark_pred.txt
    sleep 10
done
