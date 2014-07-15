#!/bin/bash

OPTION[0]="--src=largestdegree --device=0,1 --partition_method=random --queue-sizing=4.0"
#OPTION[0]="" #directed and do not mark-pred"
OPTION[1]=${OPTION[0]}" --mark-pred" #directed and mark-pred"
OPTION[2]=${OPTION[0]}" --undirected" #undirected and do not mark-pred"
OPTION[3]=${OPTION[1]}" --undirected" #undirected and mark-pred"
OPTION[4]=${OPTION[0]}" --idempotence"
OPTION[5]=${OPTION[1]}" --idempotence"
OPTION[6]=${OPTION[2]}" --idempotence"
OPTION[7]=${OPTION[3]}" --idempotence"

MARK[0]=""
MARK[1]=${MARK[0]}".mark_pred"
MARK[2]=${MARK[0]}".undir"
MARK[3]=${MARK[1]}".undir"
MARK[4]=${MARK[0]}".idempotence"
MARK[5]=${MARK[1]}".idempotence"
MARK[6]=${MARK[2]}".idempotence"
MARK[7]=${MARK[3]}".idempotence"

#put OS and Device type here
SUFFIX="ubuntu12.04.k40cx2"
EXCUTION="./bin/test_bfs_6.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    for j in 0 1 2 3 4 6
    do
        echo $EXCUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]} "> eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt"
        $EXCUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]} > eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done
