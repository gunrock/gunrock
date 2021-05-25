#!/bin/bash

APP_NAME="ss"

BIN_PREFIX="../../build/bin/"
DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB"


APP_OPTIONS="--undirected --num-runs 10 --quick"

#OUTPUT_DIR=${1:-"eval_mgpu"}
JSON_FILE=${1:-""}

NAME[0]="pokec"

GRAPH[0]="market $DATA_PREFIX/${NAME[0]}/${NAME[0]}.mtx"

for i in {0..0}
do

   #echo \
   $BIN_PREFIX$APP_NAME \
   ${GRAPH[$i]} \
   $APP_OPTIONS \
   --jsonfile=$JSON_FILE.json \
   > $JSON_FILE.output.txt
   #> /dev/null 2>&1
   #"> /dev/null 2>&1"
   #"> $OUTPUT_DIR/${NAME[$i]}.$APP_NAME.output.txt"
   #> $OUTPUT_DIR/${NAME[$i]}.$APP_NAME.output.txt
   #--jsondir=$OUTPUT_DIR \

done
