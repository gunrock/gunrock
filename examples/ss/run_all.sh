#!/bin/bash

exe_file="./bin/ss_main_10.0_x86_64"

OPTION="--undirected --num-runs=10"

#put OS and Device type here
EXCUTION=$exe_file
DATADIR="/data/gunrock_dataset/large"

for k in 0
do
    #put OS and Device type here
    SUFFIX="ubuntu16.04_TitanV"
    LOGDIR=eval/$SUFFIX
    mkdir -p $LOGDIR

    for i in kron_g500-logn21 mouse_gene soc-friendster soc-twitter-2010 enron
    do
        echo $EXCUTION market $DATADIR/$i/$i.mtx $OPTION --jsondir=$LOGDIR "> $LOGDIR/$i.txt 2>&1"
             $EXCUTION market $DATADIR/$i/$i.mtx $OPTION --jsondir=$LOGDIR  > $LOGDIR/$i.txt 2>&1
        sleep 30
    done
done

