#!/bin/bash

BASEOPTION="--src=randomize2 --queue-sizing=1.2 --iteration-num=16"
BASEFLAG=""
EXECUTION="./bin/test_sssp_8.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

OPTION[0]="" && FLAG[0]=".default"
OPTION[1]=" --undirected" && FLAG[1]=".undir"

OPTION[4]="" && FLAG[4]=".skip_pred"
OPTION[5]=" --mark-pred" && FLAG[5]=".mark_pred"

OPTION[8]="" && FLAG[8]=".32bit_VertexId"
OPTION[9]=" --64bit-SizeT" && FLAG[9]=".64bit_SizeT"
# OPTION[10]=" --64bit-VertexId" && FLAG[10]=".64bit_VertexId"

OPTION[11]="" && FLAG[11]=".DEF"
OPTION[12]=" --traversal-mode=LB" && FLAG[12]=".LB"
OPTION[13]=" --traversal-mode=TWC" && FLAG[13]=".TWC"
OPTION[14]=" --traversal-mode=LB_CULL" && FLAG[14]=".LB_CULL"
OPTION[15]=" --traversal-mode=LB_LIGHT_CULL" && FLAG[15]=".LB_LIGHT_CULL"
OPTION[16]=" --traversal-mode=LB_LIGHT" && FLAG[16]=".LB_LIGHT"

NAME[ 0]="soc-twitter-2010"  
NAME[ 1]="hollywood-2009"    
NAME[ 2]="soc-sinaweibo"     
NAME[ 3]="soc-orkut"
NAME[ 4]="soc-LiveJournal1"
NAME[ 5]="uk-2002"
NAME[ 6]="uk-2005"
NAME[ 7]="webbase-2001"
NAME[ 8]="indochina-2004"
NAME[ 9]="arabic-2005"
NAME[10]="europe_osm"
NAME[11]="asia_osm"
NAME[12]="germany_osm"
NAME[13]="road_usa"
NAME[14]="road_central"

cd ~/Projects/gunrock_dev/gunrock/tests/sssp

for d in 2 #{1..6}
do
    SUFFIX="CentOS6_6.k40cx${d}.rand"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done
    #QUEUE_SIZING=1 #$(echo "${d}+1" | bc)

    for o1 in {0..1}; do for o3 in {4..5}; do for o5 in {8..9}; do for o6 in {11..16}; do

    OPTIONS=${BASEOPTION}${OPTION[${o1}]}${OPTION[${o3}]}${OPTION[${o5}]}${OPTION[${o6}]}
    FLAGS=${BASEFLAG}${FLAG[${o1}]}${FLAG[${o3}]}${FLAG[${o5}]}${FLAG[${o6}]}

    for i in 9 #{0..14}
    do
        echo $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTIONS --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/${NAME[$i]}${FLAGS}.txt"
             $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTIONS --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/${NAME[$i]}${FLAGS}.txt
        sleep 1
    done;
    done; done; done; done
done

