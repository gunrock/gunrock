#!/bin/sh

OPTION1="" #directed and do not mark-pred"
OPTION2="--mark-pred" #directed and mark-pred"
OPTION3="--undirected" #undirected and do not mark-pred"
OPTION4="--undirected --mark-pred" #undirected and mark-pred"

#put OS and Device type here
SUFFIX="ubuntu12.04.k40c"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    echo ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION1
         ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION1 > eval/$SUFFIX/$i.$SUFFIX.dir_no_mark_pred.txt
    sleep 1
    echo ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION2
         ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION2 > eval/$SUFFIX/$i.$SUFFIX.dir_mark_pred.txt
    sleep 1
    echo ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION3
         ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION3 > eval/$SUFFIX/$i.$SUFFIX.undir_no_mark_pred.txt
    sleep 1
    echo ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION4
         ./bin/test_dobfs_6.0_x86_64 market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION4 > eval/$SUFFIX/$i.$SUFFIX.undir_mark_pred.txt
    sleep 1
done
