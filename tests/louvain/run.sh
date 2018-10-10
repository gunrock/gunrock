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

OPTIONS="--edge-value-min=1 --edge-value-range=1 --device=2 --advance-mode=ALL_EDGES --quick --queue-factor=0.1 --validation=each"
OPTIONS="$OPTIONS --neighborcomm-th=-1.0,0.00,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20"
OPTIONS="$OPTIONS --remove-self-loops=false --remove-duplicate-edges=false"
OPTIONS="$OPTIONS --pass-stats --iter-stats --unify-segments=1,0"
#OPTIONS="$OPTIONS --max-iters=10 --max-passes=10"
OPTIONS="$OPTIONS --iter-th=0.001 --pass-th=0.0001"
#OPTIONS="$OPTIONS --omp-threads=1,2,3,4,6,8,12,16,24,32 --omp-runs=5"
OPTIONS="$OPTIONS --omp-threads=16 --omp-runs=0 --num-runs=10"
#OPTIONS="$OPTIONS --1st-th=0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001"
OPTION[0]="$OPTIONS"
OPTION[1]="${OPTION[0]} --undirected" #undirected"

MARK[0]=""
MARK[1]="${MARK[0]}_undir"

#put OS and Device type here
EXCUTION=$exe_file
DATADIR="/data/gunrock_dataset/large"

NAME[ 0]="cnr-2000"
NAME[ 1]="coPapersDBLP"
NAME[ 2]="soc-LiveJournal1"
NAME[ 3]="channel-500x100x100-b050"
NAME[ 4]="uk-2002"
NAME[ 5]="europe_osm"
NAME[ 6]="rgg_n_2_24_s0"
NAME[ 7]="nlpkkt240"
NAME[ 8]="uk-2005"
NAME[ 9]="webbase-1M"
NAME[10]="webbase-2001"
NAME[11]="preferentialAttachment"
NAME[12]="caidaRouterLevel"
NAME[13]="citationCiteseer"
NAME[14]="coAuthorsDBLP"
NAME[15]="coPapersCiteseer"
NAME[16]="hollywood-2009"
NAME[17]="as-Skitter"

for k in 0
do
    #put OS and Device type here
    SUFFIX="ubuntu16.04_V100x1m4"
    LOGDIR=eval/$SUFFIX
    mkdir -p $LOGDIR

    for i in 0 1 2 3 4 5 6 9 11 12 13 14 15 16 17
    do
        for j in {0..1}
        do
            echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --jsondir=$LOGDIR "> $LOGDIR/${NAME[$i]}${MARK[$j]}.txt 2>&1"
                 $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --jsondir=$LOGDIR  > $LOGDIR/${NAME[$i]}${MARK[$j]}.txt 2>&1
            sleep 30
        done
    done
done

