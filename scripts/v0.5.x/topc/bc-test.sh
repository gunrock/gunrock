#!/bin/bash

EXEDIR=${1:-"../../build/bin"}
DATADIR=${2:-"/data/gunrock_dataset/large"}
EXECUTION="bc"
SETTING=" --src=0 --iteration-num=10"
NAME[0]="soc-LiveJournal1" && DO_A[0]="0.200"   && DO_B[0]="0.1" && T_MODE[0]="LB_CULL"
NAME[1]="soc-orkut"        && DO_A[1]="0.012"   && DO_B[1]="0.1" && T_MODE[1]="LB_CULL"
NAME[2]="hollywood-2009"   && DO_A[2]="0.006"   && DO_B[2]="0.1" && T_MODE[2]="LB_CULL"
NAME[3]="indochina-2004"   && DO_A[3]="100"     && DO_B[3]="100" && T_MODE[3]="LB_CULL"
NAME[4]="rmat_n22_e64"     && DO_A[4]="0.00001" && DO_B[4]="0.1" && T_MODE[4]="LB_CULL"
NAME[5]="rmat_n23_e32"     && DO_A[5]="0.00001" && DO_B[5]="0.1" && T_MODE[5]="LB_CULL"
NAME[6]="rmat_n24_e16"     && DO_A[6]="0.00001" && DO_B[6]="0.1" && T_MODE[6]="LB_CULL"
NAME[7]="road_usa"         && DO_A[7]="1.0"     && DO_B[7]="10"  && T_MODE[7]="TWC"
NAME[8]="rgg_n24_0.000548" && DO_A[8]="1.0"     && DO_B[8]="10"  && T_MODE[8]="TWC"

GRAPH[0]="market $DATADIR/${NAME[0]}/${NAME[0]}.mtx"
GRAPH[1]="market $DATADIR/${NAME[1]}/${NAME[1]}.mtx"
GRAPH[2]="market $DATADIR/${NAME[2]}/${NAME[2]}.mtx"
GRAPH[3]="market $DATADIR/${NAME[3]}/${NAME[3]}.mtx"
GRAPH[4]="grmat --rmat_scale=22 --rmat_edgefactor=64"
GRAPH[5]="grmat --rmat_scale=23 --rmat_edgefactor=32"
GRAPH[6]="grmat --rmat_scale=24 --rmat_edgefactor=16"
GRAPH[7]="market $DATADIR/${NAME[7]}/${NAME[7]}.mtx"
GRAPH[8]="rgg --rgg_scale=24 --rgg_threshold=0.000548"

mkdir -p eval
DEVICE="0"
for i in {0..8}
do
    echo $EXEDIR/$EXECUTION ${GRAPH[$i]} $SETTING --device=$DEVICE --jsondir=./eval/ "> ./eval/${NAME[$i]}.$EXECUTION.output.txt"
         $EXEDIR/$EXECUTION ${GRAPH[$i]} $SETTING --device=$DEVICE --jsondir=./eval/  > ./eval/${NAME[$i]}.$EXECUTION.output.txt
    sleep 1
done
