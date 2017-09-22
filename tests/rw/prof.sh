#!/bin/bash

OPTION="--undirected --quick"
MARK="gpu.Kwalklen"
EXECUTION="./bin/test_rw_9.0_x86_64"
DATADIR="/data/gunrock_dataset/large"
FORMAT="market"
DATA="hollywood-2009"
WALK="5 10 20 40 80 160"


mkdir -p ./eval/$MARK
#cd ~/Projects/gunrock_dev/gunrock/tests/bfs
for w in $WALK
do
    echo $EXECUTION $FORMAT $DATADIR/$DATA/$DATA.mtx $OPTION --walk_length=$w --jsondir=./eval/$MARK "> ./eval/$MARK/"$w"_$DATA.txt"
	$EXECUTION $FORMAT $DATADIR/$DATA/$DATA.mtx $OPTION --walk_length=$w --jsondir=./eval/$MARK > ./eval/$MARK/"$w"_$DATA.txt
done    
   
#nvprof --kernels RandomNext --metrics gld_efficiency ./bin/test_rw_9.0_x86_64 market /data/gunrock_dataset/small/chesapeake.mtx --quick --undirected --walk_length=4 --quiet
