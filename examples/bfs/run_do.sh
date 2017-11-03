#!/bin/bash

BASEOPTION="--src=0 --queue-sizing=6.5 --in-sizing=4 --iteration-num=10"
BASEFLAG=""
EXECUTION="./bin/test_bfs_8.0_x86_64"
DATADIR="/data/gunrock_dataset/large"


OPTION[0]="" && FLAG[0]=".default"
OPTION[1]=" --undirected" && FLAG[1]=".undir"

OPTION[2]="" && FLAG[2]=".normal"
OPTION[3]=" --idempotence" && FLAG[3]=".idempotence"

OPTION[4]="" && FLAG[4]=".skip_pred"
OPTION[5]=" --mark-pred" && FLAG[5]=".mark_pred"

OPTION[6]="" && FLAG[6]=".top_down"
OPTION[7]=" --direction-optimized" && FLAG[7]=".do"

OPTION[8]="" && FLAG[8]=".32bit_VertexId"
OPTION[9]=" --64bit-SizeT" && FLAG[9]=".64bit_SizeT"
OPTION[10]=" --64bit-VertexId" && FLAG[10]=".64bit_VertexId"

OPTION[11]="" && FLAG[11]=".DEF"
OPTION[12]=" --traversal-mode=LB" && FLAG[12]=".LB"
OPTION[13]=" --traversal-mode=TWC" && FLAG[13]=".TWC"
OPTION[14]=" --traversal-mode=LB_CULL" && FLAG[14]=".LB_CULL"

NAME[ 0]="soc-twitter-2010" && DO_A[ 0]="0.005" && DO_B[ 0]="0.1"
NAME[ 1]="hollywood-2009"   && DO_A[ 1]="0.006" && DO_B[ 1]="0.1"
#0.1, 0.1 for single GPU
NAME[ 2]="soc-sinaweibo"    && DO_A[ 2]="0.00005" && DO_B[ 2]="0.1"
NAME[ 3]="soc-orkut"        && DO_A[ 3]="0.012" && DO_B[ 3]="0.1"
NAME[ 4]="soc-LiveJournal1" && DO_A[ 4]="0.200" && DO_B[ 4]="0.1"
NAME[ 5]="uk-2002"          && DO_A[ 5]="16"    && DO_B[ 5]="100"
NAME[ 6]="uk-2005"          && DO_A[ 6]="10"    && DO_B[ 6]="20"
NAME[ 7]="webbase-2001"     && DO_A[ 7]="100"   && DO_B[ 7]="100"
#2, 1.2 for memory
NAME[ 8]="indochina-2004"   && DO_A[ 8]="100"   && DO_B[ 8]="100"
NAME[ 9]="arabic-2005"      && DO_A[ 9]="100"   && DO_B[ 9]="100"
NAME[10]="europe_osm"       && DO_A[10]="2.5"   && DO_B[10]="15"
NAME[11]="asia_osm"         && DO_A[11]="1.5"   && DO_B[11]="10"
NAME[12]="germany_osm"      && DO_A[12]="1.5"   && DO_B[12]="10"
NAME[13]="road_usa"         && DO_A[13]="1.0"   && DO_B[13]="10"
NAME[14]="road_central"     && DO_A[14]="1.2" && DO_B[14]="10"

GRAPH[0]="market $DATADIR/${NAME[4]}/${NAME[4]}.mtx" && tNAME[0]=${NAME[4]}
GRAPH[1]="market $DATADIR/${NAME[3]}/${NAME[3]}.mtx" && tNAME[1]=${NAME[3]}
GRAPH[2]="market $DATADIR/${NAME[1]}/${NAME[1]}.mtx" && tNAME[2]=${NAME[1]}
GRAPH[3]="market $DATADIR/${NAME[8]}/${NAME[8]}.mtx" && tNAME[3]=${NAME[8]}
GRAPH[4]="grmat --rmat_scale=22 --rmat_edgefactor=64" && tNAME[4]="rmat_n22_e64"
GRAPH[5]="grmat --rmat_scale=23 --rmat_edgefactor=32" && tNAME[5]="rmat_n23_e32"
GRAPH[6]="grmat --rmat_scale=24 --rmat_edgefactor=16" && tNAME[6]="rmat_n24_e16"
GRAPH[7]="market $DATADIR/${NAME[13]}/${NAME[13]}.mtx" && tNAME[7]=${NAME[13]}
GRAPH[8]="rgg --rgg_scale=24 --rgg_thfactor=0.00548" && tNAME[8]="rgg_n24_0.000548"

for d in 1 #{1..4}
do
    SUFFIX="ubuntu14.04_k40cx${d}"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done

    for o1 in 1; do for o2 in 3; do for o3 in 4; do for o4 in 7; do for o5 in 8; do for o6 in 14; do

    OPTIONS=${BASEOPTION}${OPTION[${o1}]}${OPTION[${o2}]}${OPTION[${o3}]}${OPTION[${o4}]}${OPTION[${o5}]}${OPTION[${o6}]}
    FLAGS=${BASEFLAG}${FLAG[${o1}]}${FLAG[${o2}]}${FLAG[${o3}]}${FLAG[${o4}]}${FLAG[${o5}]}${FLAG[${o6}]}

    for i in {0..8} #{0..14}
    do

    for a in "0.000001" "0.000002" "0.000005" "0.00001" "0.00002" "0.00005" "0.0001" "0.0002" "0.0005" "0.001" "0.002" "0.005" "0.01" "0.02" "0.05" "0.1" "0.2" "0.5" "1" "2" "5" "10" "20" "50" "100" "200" "500" "1000" "2000" "5000" "10000" "20000"
    do

    for b in "0.000001" "0.000002" "0.000005" "0.00001" "0.00002" "0.00005" "0.0001" "0.0002" "0.0005" "0.001" "0.002" "0.005" "0.01" "0.02" "0.05" "0.1" "0.2" "0.5" "1" "2" "5" "10" "20" "50" "100" "200" "500" "1000" "2000" "5000" "10000" "20000"
    do

    #for i in {0..8} #{0..14}
    #do
        echo $EXECUTION ${GRAPH[$i]} $OPTIONS --device=$DEVICE --do_a=${a} --do_b=${b} --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/${tNAME[$i]}${FLAGS}.a${a}.b${b}.txt"
             $EXECUTION ${GRAPH[$i]} $OPTIONS --device=$DEVICE --do_a=${a} --do_b=${b} --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/${tNAME[$i]}${FLAGS}.a${a}.b${b}.txt
        sleep 1
    done; done; done
    done; done; done; done; done; done
done

