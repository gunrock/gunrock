#!/bin/bash

OPTION="--undirected --idempotence --src=randomize2 --traversal-mode=LB --iteration-num=10"
MARK=".skip_pred.undir.idempotence.LB.32bitSizeT.fw"
EXECUTION="./bin/test_bfs_8.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

#cd ~/Projects/gunrock_dev/gunrock/tests/bfs

for d in 1 #{1..4}
do
    SUFFIX="ubuntu14.04_k40cx${d}"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done
    queue_sizing=$(echo "1.25 * ${d}" | bc)

    scale="21"
    for edge_factor in 1 2 4 8 16
    do
        GRAPH="grmat --rmat_scale=${scale} --rmat_edgefactor=${edge_factor}"
        echo $EXECUTION $GRAPH $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_n${scale}_${edge_factor}${MARK}.txt"
             $EXECUTION $GRAPH $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/rmat_n${scale}_${edge_factor}${MARK}.txt
    done

    queue_sizing=1
    for thfactor in "0.2" "0.3" "0.4" "0.6" "0.8"
    do
        GRAPH="rgg --rgg_scale=${scale} --rgg_thfactor=${thfactor}"   
        echo $EXECUTION $GRAPH $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rgg_n${scale}_${thfactor}${MARK}.txt"
             $EXECUTION $GRAPH $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/rgg_n${scale}_${thfactor}${MARK}.txt
    done
done

