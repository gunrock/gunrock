#!/bin/bash

APP_NAME="geo"
BIN_PREFIX="../../build/bin/"

DATA_PREFIX[0]="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/geolocation/twitter/graph"
DATA_PREFIX[1]="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/geolocation/instagram/graph"

NAME[0]="twitter"
NAME[1]="instagram"

GRAPH[0]="market ${DATA_PREFIX[0]}/${NAME[0]}.mtx --labels-file=${DATA_PREFIX[0]}/${NAME[0]}.labels"
GRAPH[1]="market ${DATA_PREFIX[1]}/${NAME[1]}.mtx --labels-file=${DATA_PREFIX[1]}/${NAME[1]}.labels"

OUTPUT_DIR=${1:-"eval_mgpu"}
NUM_GPUS=${2:-"1"}
JSON_FILE=""

GEO_ITER=${3:-"10"}
SPATIAL_ITER=${4:-"1000"}
APP_OPTIONS="--geo-iter=${GEO_ITER} --spatial-iter=${SPATIAL_ITER} --quick"

VARIANT="geo-${GEO_ITER}_spatial-${SPATIAL_ITER}"
TAG="variant:$VARIANT,num-gpus:$NUM_GPUS"

SUB_DIR=${VARIANT}
mkdir -p "$OUTPUT_DIR/$SUB_DIR"

for i in {0..1}
do
   # prepare output json file name with number of gpus for this run
   JSON_FILE="geo__${NAME[$i]}__GPU${NUM_GPUS}"	

   #echo \
   $BIN_PREFIX$APP_NAME \
   ${GRAPH[$i]} \
   $APP_OPTIONS \
   --tag=$TAG \
   --jsonfile="$OUTPUT_DIR/$SUB_DIR/$JSON_FILE.json" \
   > "$OUTPUT_DIR/$SUB_DIR/$JSON_FILE.output.txt"
done
