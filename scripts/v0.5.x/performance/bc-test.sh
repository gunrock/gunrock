#!/bin/bash

EXEDIR="../../../build/bin"
EXECUTION="bc"

DATADIR="/data/gunrock_dataset/large"
DATAHUGE="/data/gunrock_dataset/huge"

SETTING[0]=" --src=0 --iteration-num=10 --traversal-mode=LB_CULL --in-sizing=1.1 --queue-sizing=1.2 --quick"
SETTING[1]=" --src=largestdegree --iteration-num=10 --traversal-mode=LB_CULL --in-sizing=1.1 --queue-sizing=1.2 --quick"

NAME[0]="cit-Patents"
NAME[1]="soc-LiveJournal1"
NAME[2]="soc-twitter-2010"
NAME[3]="uk-2002"
NAME[4]="uk-2005"
NAME[5]="kron_g500-logn21"
NAME[6]="twitter_rv.net"
NAME[7]="rmat_n24_e16"

GRAPH[0]="market $DATADIR/${NAME[0]}/${NAME[0]}.mtx"
GRAPH[1]="market $DATADIR/${NAME[1]}/${NAME[1]}.mtx"
GRAPH[2]="market $DATADIR/${NAME[2]}/${NAME[2]}.mtx"
GRAPH[3]="market $DATADIR/${NAME[3]}/${NAME[3]}.mtx"
GRAPH[4]="market $DATADIR/${NAME[4]}/${NAME[4]}.mtx"
GRAPH[5]="market $DATADIR/${NAME[5]}/${NAME[5]}.mtx"
GRAPH[6]="market $DATAHUGE/${NAME[6]}/${NAME[6]}.mtx --64bit-SizeT"
GRAPH[7]="grmat --rmat_scale=24 --rmat_edgefactor=16"

mkdir -p eval
DEVICE="0"
for j in {0..1}
do
for i in {0..8}
do
    echo $EXECUTION ${GRAPH[$i]} ${SETTING[$j]} --device=$DEVICE --jsondir=./eval/
    $EXEDIR/$EXECUTION ${GRAPH[$i]} ${SETTING[$j]} --device=$DEVICE --jsondir=./eval/ > ./eval/${NAME[$i]}.$EXECUTION.output.txt
    sleep 1
done
done
