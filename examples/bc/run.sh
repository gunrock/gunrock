#!/bin/bash

OPTION[0]="--src=largestdegree --device=0 --partition-method=random"

MARK[0]=""

#get all execution files in ./bin
files=./bin/*
#split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
exe_file=${arr[0]}
#iterate over all file names to get the largest version number
for x in $arr
do
    output=$(grep -o "[0-9]\.[0-9]" <<<"$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

#put OS and Device type here
SUFFIX="ubuntu16.04_TitanV"
EXECUTION="./bin/test_bc_9.2_x86_64" #$exe_file
DATADIR="/data/gunrock_dataset/large"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    for j in 0
    do
        echo $EXECUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]} "> eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt"
             $EXECUTION market $DATADIR/$i/$i.mtx ${OPTION[$j]}  > eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done

OPTION[0]="--src=-1 --device=0 --partition-method=random"
MARK[0]=""

for i in chesapeake test_bc
do
    for j in 0
    do
        echo $EXECUTION market $DATADIR/../small/$i.mtx ${OPTION[$j]} "> eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt"
             $EXECUTION market $DATADIR/../small/$i.mtx ${OPTION[$j]}  > eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done
