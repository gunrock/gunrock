#!/bin/bash

OPTION="--undirected --quick --normalized --traversal-mode=0 --iteration-num=10"
MARK=".undir.32bitSizeT.nocomp"
EXECUTION="./bin/test_pr_7.5_x86_64"
DATADIR="/data/gunrock_dataset/large"

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

for d in {1..4}
do
    SUFFIX="ubuntu14_04.k40cx${d}_rand"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done

    for i in {0..14}
    do
        echo $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTION --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/${NAME[$i]}${MARK}.txt"
             $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTION --device=$DEVICE --jsondir=./eval/$SUFFIX > ./eval/$SUFFIX/${NAME[$i]}${MARK}.txt
        sleep 1
    done
done

