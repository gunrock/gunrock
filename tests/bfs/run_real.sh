#!/bin/bash

#OPTION="--undirected --src=largestdegree --traversal-mode=LB_CULL --idempotence --queue-sizing=7.5 --in-sizing=4 --iteration-num=10"
OPTION="--undirected --src=largestdegree --traversal-mode=LB_CULL --idempotence --in-sizing=1 --iteration-num=10"
#queue-sizing at least 1.2 * num_gpus

MARK=".skip_pred.undir.idempotence.LB_CULL.32bitSizeT.fw"
EXECUTION="./bin/test_bfs_7.5_x86_64"
DATADIR="../../dataset/large"

NAME[ 0]="soc-twitter-2010" && DO_A[ 0]="0.005" && DO_B[ 0]="0.1"
NAME[ 1]="hollywood-2009"   && DO_A[ 1]="0.006" && DO_B[ 1]="0.1"
#0.1, 0.1 for single GPU
NAME[ 2]="soc-sinaweibo"    && DO_A[ 2]="0.00005" && DO_B[ 2]="0.1"
#NAME[ 2]="soc-sinaweibo"    && DO_A[ 2]="0.00002" && DO_B[ 2]="0.1"
#7.3, 4 for memory, do_a = 0.00002 for 5 and 6 GPUs
NAME[ 3]="soc-orkut"        && DO_A[ 3]="0.012" && DO_B[ 3]="0.1"
NAME[ 4]="soc-LiveJournal1" && DO_A[ 4]="0.200" && DO_B[ 4]="0.1"
#NAME[ 4]="soc-LiveJournal1" && DO_A[ 4]="0.140" && DO_B[ 4]="0.1"
#queue-sizing=7.5 do_a=0.14 for 5 & 6 GPUs
NAME[ 5]="uk-2002"          && DO_A[ 5]="16"    && DO_B[ 5]="100"
NAME[ 6]="uk-2005"          && DO_A[ 6]="10"    && DO_B[ 6]="20"
NAME[ 7]="webbase-2001"     && DO_A[ 7]="100"   && DO_B[ 7]="100"
#1.2, 1 for memory, 2.5 for 5 & 6 GPUs
NAME[ 8]="indochina-2004"   && DO_A[ 8]="100"   && DO_B[ 8]="100"
NAME[ 9]="arabic-2005"      && DO_A[ 9]="100"   && DO_B[ 9]="100"
NAME[10]="europe_osm"       && DO_A[10]="2.5"   && DO_B[10]="15"
NAME[11]="asia_osm"         && DO_A[11]="1.5"   && DO_B[11]="10"
NAME[12]="germany_osm"      && DO_A[12]="1.5"   && DO_B[12]="10"
NAME[13]="road_usa"         && DO_A[13]="1.0"   && DO_B[13]="10"
NAME[14]="road_central"     && DO_A[14]="1.2" && DO_B[14]="10"

cd ~/Projects/gunrock_dev/gunrock/tests/bfs

for d in {1..6}
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

    queue_sizing=2.5
    if [ "$d" -lt "5" ]; then
        queue_sizing=1.2
    fi

    for i in 7 #{0..14}
    do
        echo $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTION --queue-sizing=${queue_sizing}  --device=$DEVICE --do_a=${DO_A[$i]} --do_b=${DO_B[$i]} --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/${NAME[$i]}${MARK}.txt"
             $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTION --queue-sizing=${queue_sizing}  --device=$DEVICE --do_a=${DO_A[$i]} --do_b=${DO_B[$i]} --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/${NAME[$i]}${MARK}.txt
        sleep 1
    done
done

