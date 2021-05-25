#!/bin/bash

APP_NAME="ss"

BIN_PREFIX="../../build/bin/"
DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB"


APP_OPTIONS="--undirected --num-runs 10 --quick"

OUTPUT_DIR=${1:-"eval_mgpu"}
NUM_GPUS=${2:-"1"}
JSON_FILE=""

TIMESTAMP=`date '+%Y-%m-%d_%H:%M:%S'`
TAG=$TIMESTAMP

NAME[0]="pokec"

GRAPH[0]="market $DATA_PREFIX/${NAME[0]}/${NAME[0]}.mtx"

for i in {0..0}
do
   # prepare output json file name with number of gpus for this run
   JSON_FILE="${APP_NAME}__${NAME[$i]}__GPU${NUM_GPUS}"	

   #echo \
   $BIN_PREFIX$APP_NAME \
   ${GRAPH[$i]} \
   $APP_OPTIONS \
   --tag=$TAG \
   --jsonfile="$OUTPUT_DIR/$JSON_FILE.json" \
   > "$OUTPUT_DIR/$JSON_FILE.output.txt"
done
