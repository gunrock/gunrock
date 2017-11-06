#!/bin/bash

OPTION="--undirected --idempotence --src=randomize2 --traversal-mode=LB_CULL --in-sizing=5 --iteration-num=10 --do_a=0.0008 --do_b=0.08"
MARK=".skip_pred.undir.idempotence.LB_CULL.32bitSizeT.fw"
EXECUTION="./bin/test_bfs_8.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

cd ~/Projects/gunrock_dev/gunrock/tests/bfs

for d in {1..4}
do
    SUFFIX="CentOS7.2_p100x${d}_rand"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done
    queue_sizing=$(echo "1.25 * ${d}" | bc)

    rmat_method="grmat"
    if [ "$d" -eq "1" ]; then
        rmat_method="rmat"
    fi
    
    scale="24"
    edge_factor="32"
    echo $EXECUTION ${rmat_method} --rmat_scale=${scale} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_n${scale}_${edge_factor}${MARK}.txt"
         $EXECUTION ${rmat_method} --rmat_scale=${scale} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/rmat_n${scale}_${edge_factor}${MARK}.txt
   
    rmat_method="grmat" 
    num_nodes=$(echo "(2^19) * ${d}" | bc)
    edge_factor="256"
    if [ "$d" -eq "8" ]; then
        edge_factor="255"
    fi
    echo $EXECUTION ${rmat_method} --rmat_nodes=${num_nodes} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_${num_nodes}_${edge_factor}${MARK}.txt"
         $EXECUTION ${rmat_method} --rmat_nodes=${num_nodes} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/rmat_${num_nodes}_${edge_factor}${MARK}.txt

    scale="19"
    edge_factor=$(echo "256 * ${d}" | bc)
    if [ "$d" -eq "8" ]; then
        edge_factor=$(echo "${edge_factor} - 1" | bc)
    fi
    echo $EXECUTION ${rmat_method} --rmat_scale=${scale} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/rmat_n${scale}_${edge_factor}${MARK}.txt"
         $EXECUTION ${rmat_method} --rmat_scale=${scale} --rmat_edgefactor=${edge_factor} $OPTION --queue-sizing=${queue_sizing} --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/rmat_n${scale}_${edge_factor}${MARK}.txt
 
done

