#!/bin/bash

OPTION[0]="--src=largestdegree --device=0 --partition_method=random"

MARK[0]=""

#put OS and Device type here
SUFFIX="ubuntu12.04.k40c"
EXCUTION="./bin/test_bc_6.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    for j in 0
    do
        echo $EXCUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]} "> eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt"
             #$EXCUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]}  > eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done

OPTION[0]="--src=-1 --device=0 --partition_method=random"
MARK[0]=""

for i in chesapeake test_bc
do
    for j in 0
    do
        echo $EXCUTION market $DATADIR/../small/$i.mtx ${OPTION[$j]} "> eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt"
             #$EXCUTION market $DATADIR/../small/$i.mtx ${OPTION[$j]}  > eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done
