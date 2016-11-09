#!/bin/bash

OPTION="--idempotence --src=largestdegree --traversal-mode=LB_CULL --direction-optimized --queue-sizing=6.5 --in-sizing=4 --iteration-num=10 --do_a=0.00001 --do_b=0.1"
MARK=".skip_pred.idempotence.LB_CULL.32bitSizeT.do"
EXECUTION="./bin/test_bfs_8.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

for d in {1..2}
do
    SUFFIX="ubuntu14.04_k40cx${d}_rand"
    mkdir -p eval/$SUFFIX
    #DEVICE="2"
    #if [ "$d" -eq "2" ]; then
    #    DEVICE=$DEVICE",3"
    #fi

    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done

    for i in {20..25}
    do
        max_j=$(echo "29-${i}" | bc)
        for j in {4..9}
        do
            edge_factor=$(echo "2^${j}" | bc)
            if [ "$j" -le $((29-${i})) ]; then 
                echo $EXECUTION grmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt"
                     $EXECUTION grmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --device=$DEVICE --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt
                sleep 1
            fi
        done
    done
    
    if [ "$d" -eq "1" ]; then
        for i in {20..25}
        do  
            edge_factor=$(echo "2^( 29 - ${i} )" | bc)
            echo $EXECUTION rmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt"
                 $EXECUTION rmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --device=$DEVICE --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt
        done
    fi 
done

