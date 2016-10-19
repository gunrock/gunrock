#!/bin/bash

OPTION="--src=largestdegree --traversal-mode=LB_CULL --in-sizing=2.2 --iteration-num=10"
#OPTION="--src=largestdegree --traversal-mode=LB_CULL --in-sizing=0 --iteration-num=10"
MARK=".skip_pred.LB_CULL.32bitSizeT"
EXECUTION="./bin/test_bc_7.5_x86_64"
DATADIR="../../dataset/large"

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
NAME[14]="road_central"     && DO_A[14]="1.2"   && DO_B[14]="10"

cd ~/Projects/gunrock_dev/gunrock/tests/bc

for d in 6 #{5..6}
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

    QUEUE_SIZING=1.2 #$(echo "${d} * 1.21" | bc)
    for i in 0  #{0..9}
    do
        echo $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTION --queue-sizing=$QUEUE_SIZING --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/${NAME[$i]}${MARK}.txt"
             $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTION --queue-sizing=$QUEUE_SIZING --device=$DEVICE --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/${NAME[$i]}${MARK}.txt
        sleep 1
    done
done

