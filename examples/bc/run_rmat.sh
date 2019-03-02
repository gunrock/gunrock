#!/bin/bash

OPTION="--src=largestdegree --traversal-mode=LB_CULL --in-sizing=1 --iteration-num=10"
MARK=".skip_pred.LB_CULL.32bitSizeT"
EXECUTION="./bin/test_bc_7.5_x86_64"
DATADIR="../../dataset/large"

cd ~/Projects/gunrock_dev/gunrock/tests/bc

for d in {4..6}
do
    SUFFIX="CentOS6_6.k40cx${d}.rand"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done
    queue_sizing=$(echo "${d} * 1.21" | bc)
    for i in 25 #{20..25}
    do
        max_j=$(echo "29-${i}" | bc)
        for j in {4..9}
        do
            edge_factor=$(echo "2^${j}" | bc)
            if [ "$j" -le $((29-${i})) ]; then 
                echo $EXECUTION grmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt"
                     $EXECUTION grmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt
                sleep 1
            fi
        done
    done
    
    if [ "$d" -eq "1" ]; then
        for i in {20..25}
        do  
            edge_factor=$(echo "2^( 29 - ${i} )" | bc)
            echo $EXECUTION rmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt"
                 $EXECUTION rmat --rmat_scale=${i} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/rmat_n${i}_${edge_factor}${MARK}.txt
        done
    fi 
done

