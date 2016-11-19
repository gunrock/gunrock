#!/bin/bash

EXEDIR="../../../../gunrock_build/bin"
EXECUTION="sssp"
DATADIR="/data/Compare/topc-datasets"
SETTING=" --src=0 --undirected --iteration-num=10"
NAME[0]="soc-LiveJournal1" && DO_A[0]="0.200" && DO_B[0]="0.1" && T_MODE[0]="LB_CULL"
NAME[1]="soc-orkut" && DO_A[1]="0.012" && DO_B[1]="0.1" && T_MODE[1]="LB_CULL"
NAME[2]="hollywood-2009" && DO_A[2]="0.006" && DO_B[2]="0.1" && T_MODE[2]="LB_CULL"
NAME[3]="indochina-2004" && DO_A[3]="100" && DO_B[3]="100" && T_MODE[3]="LB_CULL"
NAME[4]="rmat_n22_e64" && DO_A[4]="0.00001" && DO_B[4]="0.1" && T_MODE[4]="LB_CULL"
NAME[5]="rmat_n23_e32" && DO_A[5]="0.00001" && DO_B[5]="0.1" && T_MODE[5]="LB_CULL"
NAME[6]="rmat_n24_e16" && DO_A[6]="0.00001" && DO_B[6]="0.1" && T_MODE[6]="LB_CULL"
NAME[7]="road_usa" && DO_A[7]="1.0" && DO_B[7]="10" && T_MODE[7]="TWC"
NAME[8]="rgg_n24_0.000548" && DO_A[8]="1.0" && DO_B[8]="10" && T_MODE[8]="TWC"

mkdir -p eval
DEVICE="0"
for i in {0..8}
do
    echo $EXECUTION ${NAME[$i]} $SETTING
    $EXEDIR/$EXECUTION market $DATADIR/${NAME[$i]}.mtx $SETTING --device=$DEVICE --traversal-mode=${T_MODE[$i]} --jsondir=./eval/ > ./eval/${NAME[$i]}.$EXECUTION.output.txt
    sleep 1
done
