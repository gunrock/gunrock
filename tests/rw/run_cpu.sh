#!/bin/bash

OPTION="--undirected --walk_length=80"
MARK="cpu.gpu.walklen80"
EXECUTION="./bin/test_rw_9.0_x86_64"
DATADIR="/data/gunrock_dataset/large"
FORMAT="market"
DATA="hollywood-2009 soc-LiveJournal1 indochina-2004 soc-orkut road_usa"


mkdir -p ./eval/$MARK
#cd ~/Projects/gunrock_dev/gunrock/tests/bfs
for graph in $DATA
do
    echo $EXECUTION $FORMAT $DATADIR/$graph/$graph.mtx $OPTION --jsondir=./eval/$MARK "> ./eval/$MARK/$graph.txt"
	$EXECUTION $FORMAT $DATADIR/$graph/$graph.mtx $OPTION --jsondir=./eval/$MARK > ./eval/$MARK/$graph.txt
done    
   

