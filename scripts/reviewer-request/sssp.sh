#!/bin/bash

EXEDIR="../../../v.5_build/bin"
EXECUTION="sssp"
DATADIR="../../dataset/large"

# General settings
SETTING="--iteration-num=10"
DEVICE="0"

# Traversal settings
T_MODE[0]="TWC" && T_MODE[1]="LB" && T_MODE[2]="LB_CULL"

# Source vertex = 0, largestdegree or random
SRC[0]="0" && SRC[1]="largestdegree" && SRC[2]="random"

# Undirected true or false
UNDIR[0]="--undirected" && UNDIR[1]=""

NAME[0]="orkut-dir"
# NAME[1]="road_usa"
# NAME[2]="rmat_n25_e8"

GRAPH[0]="market $DATADIR/${NAME[0]}/${NAME[0]}.mtx"
# GRAPH[1]="market $DATADIR/${NAME[1]}/${NAME[1]}.mtx"
# GRAPH[2]="grmat --rmat_scale=25 --rmat_edgefactor=8"

mkdir -p sssp

for d in {0..2}
do
    for s in {0..2}
    do
        for u in {0..1}
        do
            for l in {0..2}
            do
                echo $EXECUTION ${GRAPH[$d]} --src=${SRC[$s]} $SETTING ${UNDIR[$u]} --device=$DEVICE --traversal-mode=${T_MODE[$l]} --jsondir=./sssp/
                $EXEDIR/$EXECUTION ${GRAPH[$d]} --src=${SRC[$s]} $SETTING ${UNDIR[$u]} --device=$DEVICE --traversal-mode=${T_MODE[$l]} --jsondir=./sssp/ >> ./sssp/${NAME[$d]}.$EXECUTION.output.txt
                sleep 1
            done
        done
    done
done
