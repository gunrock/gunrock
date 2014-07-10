#!/bin/bash

OPTION[0]="--src=largestdegree --device=0,1,2,3 --partition_method=random --queue-sizing=8.0"
OPTION[1]=${OPTION[0]}" --undirected" #undirected and do not mark-pred"
OPTION[2]=${OPTION[0]}" --mark-path" #undirected and mark-pred"
OPTION[3]=${OPTION[1]}" --mark-path"

MARK[0]=""
MARK[1]=${MARK[0]}".undir"
MARK[2]=${MARK[0]}".mark_path"
MARK[3]=${MARK[1]}".mark_path"

#put OS and Device type here
SUFFIX="ubuntu12.04.k40cx4"
EXCUTION="./bin/test_sssp_6.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    for j in 0 1 2 3
    do  
        echo $EXCUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]} "> eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt"
        $EXCUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]} > eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done


