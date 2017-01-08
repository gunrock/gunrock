#!/bin/bash

EXEDIR="../../../gunrock_build/bin"
EXECUTION="bfs"
DATADIR="../large"
SETTING=" --src=0 --undirected --idempotence --queue-sizing=6.5 --in-sizing=4 --iteration-num=10 --direction-optimized"
NAME[0]="soc-orkut" && DO_A[0]="0.012" && DO_B[0]="0.1" && T_MODE[0]="LB_CULL"
NAME[1]="hollywood-2009" && DO_A[1]="0.006" && DO_B[1]="0.1" && T_MODE[1]="LB_CULL"
NAME[2]="indochina-2004" && DO_A[2]="100" && DO_B[2]="100" && T_MODE[2]="LB_CULL"
NAME[3]="kron_g500-logn21" && DO_A[3]="0.00001" && DO_B[3]="0.1" && T_MODE[3]="LB_CULL"
NAME[4]="roadNet-CA" && DO_A[4]="1.0" && DO_B[4]="10" && T_MODE[4]="TWC"

mkdir -p eval
DEVICE="0"
for i in {0..4}
do
    echo $EXECUTION ${NAME[$i]} $SETTING
    nvprof $EXEDIR/$EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $SETTING --device=$DEVICE --traversal-mode=${T_MODE[$i]} --do_a=${DO_A[$i]} --do_b=${DO_B[$i]} --jsondir=./eval/ > ./eval/nvprof.${NAME[$i]}.$EXECUTION.output.txt
    sleep 1
done
echo $EXECUTION rgg_24 $SETTING
#$EXEDIR/$EXECUTION rgg --rgg_scale=24 $SETTING --device=$DEVICE --traversal-mode=TWC --do_a=${DO_A[4]} --do_b=${DO_B[4]} --jsondir=./eval/ > ./eval/rgg_24.$EXECUTION.output.txt


