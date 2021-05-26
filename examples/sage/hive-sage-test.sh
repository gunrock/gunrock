#!/bin/bash

APP_NAME="sage"

BIN_PREFIX="../../build/bin/"
DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB"
DATA_PREFIX2="/home/u00u7u37rw7AjJoA4e357/data/gunrock/gunrock_dataset/mario-2TB/large"

APP_OPTIONS="--undirected --batch-size 16384 --quick"

OUTPUT_DIR=${1:-"eval_mgpu"}
NUM_GPUS=${2:-"1"}
JSON_FILE=""

TAG="num-gpus:$NUM_GPUS"

NAME[0]="pokec"
NAME[1]="dir_gs_twitter"
NAME[2]="europe_osm"

GRAPH[0]="market $DATA_PREFIX/${NAME[0]}/${NAME[0]}.mtx"
GRAPH[1]="market $DATA_PREFIX/graphsearch/${NAME[1]}.mtx"
GRAPH[2]="market $DATA_PREFIX2/${NAME[2]}/${NAME[2]}.mtx"

for i in {0..2}
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
