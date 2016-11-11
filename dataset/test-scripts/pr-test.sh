#!/bin/bash

EXEDIR="../../../gunrock_build/bin"
EXECUTION="pr"
DATADIR="../large"
SETTING=" --quick --iteration-num=10"
NAME[0]="soc-orkut" 
NAME[1]="hollywood-2009"
NAME[2]="indochina-2004"
NAME[3]="kron_g500-logn21"
NAME[4]="roadNet-CA"

mkdir -p eval
DEVICE="0"
for i in {0..4}
do
    echo $EXECUTION ${NAME[$i]} $SETTING
    $EXEDIR/$EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $SETTING --device=$DEVICE --jsondir=./eval/ > ./eval/${NAME[$i]}.$EXECUTION.output.txt
    sleep 1
done
echo $EXECUTION rgg_24 $SETTING
$EXEDIR/$EXECUTION rgg --rgg_scale=24 $SETTING --device=$DEVICE --jsondir=./eval/ > ./eval/rgg_24.$EXECUTION.output.txt
