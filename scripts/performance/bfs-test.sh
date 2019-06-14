#!/bin/bash

EXEDIR="../../../build/bin"
EXECUTION="bfs"

DATADIR="/data/gunrock_dataset/large"
DATAHUGE="/data/gunrock_dataset/huge"

SETTING[0]=" --src=0 --undirected --idempotence --queue-sizing=6.5 --in-sizing=4 --iteration-num=10 --direction-optimized"
SETTING[1]=" --src=0 --idempotence --queue-sizing=6.5 --in-sizing=4 --iteration-num=10 --direction-optimized"


NAME[0]="cit-Patents"           && DO_A[0]="2"          && DO_B[0]="0.00005"    && T_MODE[0]="LB_CULL"  && DEVICE[0]="0"
NAME[1]="soc-LiveJournal1"      && DO_A[1]="0.200"      && DO_B[1]="0.1"        && T_MODE[1]="LB_CULL"  && DEVICE[1]="0"
NAME[2]="soc-twitter-2010"      && DO_A[2]="0.005"      && DO_B[2]="0.1"        && T_MODE[2]="LB_CULL"  && DEVICE[2]="0"
NAME[3]="uk-2002"               && DO_A[3]="16"         && DO_B[3]="100"        && T_MODE[3]="LB_CULL"  && DEVICE[3]="0"
NAME[4]="uk-2005"               && DO_A[4]="10"         && DO_B[4]="20"         && T_MODE[4]="LB_CULL"  && DEVICE[4]="0"
NAME[5]="kron_g500-logn21"      && DO_A[5]="0.00001"    && DO_B[5]="0.1"        && T_MODE[5]="LB_CULL"  && DEVICE[5]="0"
NAME[6]="twitter_rv.net"        && DO_A[6]="0.005"      && DO_B[6]="0.1"        && T_MODE[6]="LB_CULL"  && DEVICE[6]="0"
NAME[7]="rmat_n24_e16"          && DO_A[7]="0.00001"    && DO_B[7]="0.1"        && T_MODE[7]="LB_CULL"  && DEVICE[7]="0"

GRAPH[0]="market $DATADIR/${NAME[0]}/${NAME[0]}.mtx"
GRAPH[1]="market $DATADIR/${NAME[1]}/${NAME[1]}.mtx"
GRAPH[2]="market $DATADIR/${NAME[2]}/${NAME[2]}.mtx"
GRAPH[3]="market $DATADIR/${NAME[3]}/${NAME[3]}.mtx"
GRAPH[4]="market $DATADIR/${NAME[4]}/${NAME[4]}.mtx"
GRAPH[5]="market $DATADIR/${NAME[5]}/${NAME[5]}.mtx"
GRAPH[6]="market $DATAHUGE/${NAME[6]}/${NAME[6]}.mtx --64bit-SizeT"
GRAPH[7]="grmat --rmat_scale=24 --rmat_edgefactor=16"

mkdir -p eval

for j in {0..1}
do
for i in {0..7}
do
    echo $EXECUTION ${GRAPH[$i]} ${SETTING[$j]} --device=${DEVICE[$i]} --traversal-mode=${T_MODE[$i]} --do_a=${DO_A[$i]} --do_b=${DO_B[$i]} --jsondir=./eval/
    $EXEDIR/$EXECUTION ${GRAPH[$i]} ${SETTING[$j]} --device=${DEVICE[$i]} --traversal-mode=${T_MODE[$i]} --do_a=${DO_A[$i]} --do_b=${DO_B[$i]} --jsondir=./eval/ > ./eval/${NAME[$i]}.$EXECUTION.output.txt
    sleep 1
done
done
