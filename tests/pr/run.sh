#!/bin/sh

OPTION1="--undirected" #undirected
OPTION2="--quick" #quick running without CPU reference algorithm, if you want to test CPU reference algorithm, delete $OPTION2 in some lines. Warning: for large data this can take a long time.

#put OS and Device type here
SUFFIX="ubuntu12.04.k40c"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 
do
    echo ./bin/test_pr_6.0_x86_64 market ../../dataset/large/$i/$i.mtx $OPTION1
         ./bin/test_pr_6.0_x86_64 market ../../dataset/large/$i/$i.mtx $OPTION1 > eval/$SUFFIX/$i.$SUFFIX.undir.txt
    sleep 1
done

for i in soc-LiveJournal1 kron_g500-logn21
do
    echo ./bin/test_pr_6.0_x86_64 market ../../dataset/large/$i/$i.mtx $OPTION1 $OPTION2
         ./bin/test_pr_6.0_x86_64 market ../../dataset/large/$i/$i.mtx $OPTION1 $OPTION2 > eval/$SUFFIX/$i.$SUFFIX.undir_quick.txt
    sleep 1
done

echo ./bin/test_pr_6.0_x86_64 market ../../dataset/large/webbase-1M/webbase-1M.mtx $OPTION2
         ./bin/test_pr_6.0_x86_64 market ../../dataset/large/webbase-1M/webbase-1M.mtx $OPTION2 > eval/$SUFFIX/webbase-1M.$SUFFIX.quick.txt
