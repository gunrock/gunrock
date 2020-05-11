#!/bin/bash

EXEDIR="../../../v.5_build/bin"
EXECUTION="bfs"
DATADIR="../../dataset/large"

# General settings
SETTING="--idempotence --queue-sizing=6.5 --in-sizing=4 --iteration-num=10 --quick"
DEVICE="0"

# Traversal settings
T_MODE[0]="TWC" && T_MODE[1]="LB" && T_MODE[2]="LB_CULL"

# Source vertex = 0, largestdegree or random
SRC[0]="0" && SRC[1]="largestdegree" && SRC[2]="random"

# Undirected true or false
UNDIR[0]="--undirected" && UNDIR[1]=""

NAME[0]="orkut-dir"           && DO_A[0]="5"          && DO_B[0]="10"
# NAME[1]="road_usa"            && DO_A[1]="1.0"        && DO_B[1]="10"
# NAME[2]="rmat_n25_e8"         && DO_A[2]="0.00001"    && DO_B[2]="0.1"

GRAPH[0]="market $DATADIR/${NAME[0]}/${NAME[0]}.mtx"
# GRAPH[1]="market $DATADIR/${NAME[1]}/${NAME[1]}.mtx"
# GRAPH[2]="grmat --rmat_scale=25 --rmat_edgefactor=8"

mkdir -p bfs

for d in {0..2}
do
    for s in {0..2}
    do
        for u in {0..1}
        do
            for l in {0..2}
            do
                echo $EXECUTION ${GRAPH[$d]} --src=${SRC[$s]} $SETTING ${UNDIR[$u]} --device=$DEVICE --traversal-mode=${T_MODE[$l]} --do_a=${DO_A[$d]} --do_b=${DO_B[$d]} --jsondir=./bfs/
                $EXEDIR/$EXECUTION ${GRAPH[$d]} --src=${SRC[$s]} $SETTING ${UNDIR[$u]} --device=$DEVICE --traversal-mode=${T_MODE[$l]} --do_a=${DO_A[$d]} --do_b=${DO_B[$d]} --jsondir=./bfs/ >> ./bfs/${NAME[$d]}.$EXECUTION.output.txt
                sleep 1
            done
        done
    done
done
