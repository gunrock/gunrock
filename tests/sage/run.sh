#!/bin/bash

#get all execution files in ./bin
files=(./bin/*)
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

OPTIONS="--device=2"
OPTIONS="$OPTIONS --omp-threads=32 --num-runs=10"
OPTIONS="$OPTIONS --feature-column=64,128"
OPTIONS="$OPTIONS --num-children-per-source=10,25,100"
OPTIONS="$OPTIONS --batch-size=512,1024,2048,4096,8192,16384"

OPTION[0]="$OPTIONS"
OPTION[1]="${OPTION[0]} --undirected" #undirected"

MARK[0]=""
MARK[1]="${MARK[0]}_undir"

#put OS and Device type here
EXCUTION=$exe_file
DATADIR="/data/hive_datasets"

NAME[ 0]="pokec"
NAME[ 1]="amazon"
NAME[ 2]="flickr"
NAME[ 3]="twitter"
NAME[ 4]="cit-Patents"

for k in 0
do
    #put OS and Device type here
    SUFFIX="ubuntu18.04_V100x1"
    LOGDIR=eval/$SUFFIX
    mkdir -p $LOGDIR

    for i in {0..4} ##0 1 2 3 4 5 6 9 11 12 13 14 15 16 17
    do
        for j in 1 #{0..1}
        do
            echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --jsondir=$LOGDIR "> $LOGDIR/${NAME[$i]}${MARK[$j]}.txt 2>&1"
                 $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --jsondir=$LOGDIR  > $LOGDIR/${NAME[$i]}${MARK[$j]}.txt 2>&1
            sleep 10
        done
    done
done

