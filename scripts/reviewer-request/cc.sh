#!/bin/bash

EXEDIR="../../../v.5_build/bin"
EXECUTION="cc"
DATADIR="../../dataset/large"

# General settings
SETTING="--iteration-num=10 --in-sizing=1.1 --quick"
DEVICE="1"

NAME[0]="orkut-dir"
# NAME[1]="road_usa"
# NAME[2]="rmat_n25_e8"

GRAPH[0]="market $DATADIR/${NAME[0]}/${NAME[0]}.mtx"
# GRAPH[1]="market $DATADIR/${NAME[1]}/${NAME[1]}.mtx"
# GRAPH[2]="grmat --rmat_scale=25 --rmat_edgefactor=8"

mkdir -p cc

for d in {0..2}
do          
    echo $EXECUTION ${GRAPH[$d]} $SETTING --device=$DEVICE --jsondir=./cc/
    $EXEDIR/$EXECUTION ${GRAPH[$d]} $SETTING --device=$DEVICE --jsondir=./cc/ >> ./cc/${NAME[$d]}.$EXECUTION.output.txt
    sleep 1
done
