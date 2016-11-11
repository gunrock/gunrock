#!/bin/bash

EXEDIR="../../../gunrock_build/bin"
EXECUTION="sssp"
DATADIR="../large"
SETTING=" --src=0 --undirected --iteration-num=10"
NAME[0]="soc-orkut" && T_MODE[0]="LB_CULL"
NAME[1]="hollywood-2009" && T_MODE[1]="LB_CULL"
NAME[2]="indochina-2004" && T_MODE[2]="LB_CULL"
NAME[3]="kron_g500-logn21" && T_MODE[3]="LB_CULL"
NAME[4]="roadNet-CA" && T_MODE[4]="TWC"

mkdir -p eval
DEVICE="0"
for i in {0..4}
do
    echo $EXECUTION ${NAME[$i]} $SETTING
    $EXEDIR/$EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $SETTING --device=$DEVICE --traversal-mode=${T_MODE[$i]} --jsondir=./eval/ > ./eval/${NAME[$i]}.$EXECUTION.output.txt
    sleep 1
done
echo $EXECUTION rgg_24 $SETTING
$EXEDIR/$EXECUTION rgg --rgg_scale=24 $SETTING --device=$DEVICE --traversal-mode=TWC --jsondir=./eval/ > ./eval/rgg_24.$EXECUTION.output.txt
