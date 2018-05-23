#!/bin/bash

OPTION="--undirected --quick"
MARK="gpu.Kwalklen"
EXECUTION="./bin/test_rw_9.0_x86_64"
DATADIR="/data/gunrock_dataset/large"
FORMAT="market"
DATA="hollywood-2009"
WALK="5 10 20 40 80 160"

mkdir -p ./evaluation/$MARK
DIR="./evaluation/$MARK"

#cd ~/Projects/gunrock_dev/gunrock/tests/bfs
for w in $WALK
do
    echo $EXECUTION $FORMAT $DATADIR/$DATA/$DATA.mtx $OPTION --walk_length=$w --jsondir=$DIR "> $DIR/"$w"_$DATA.txt"
	$EXECUTION $FORMAT $DATADIR/$DATA/$DATA.mtx $OPTION --walk_length=$w --jsondir=$DIR > $DIR/"$w"_$DATA.txt
done    
   

