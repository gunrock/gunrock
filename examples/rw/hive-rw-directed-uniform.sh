#!/bin/bash
## random walk: directed, uniform

APP_NAME="rw"

BIN_PREFIX="../../build/bin/"
DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/graphsearch"

DATA1="dir_gs_twitter" #.mtx
#DATA2="gs_twitter.values"

WALK_MODE="0" #0, 1, or 2
APP_OPTIONS="--walk-mode ${WALK_MODE} --walk-length 128 --undirected=false --store-walks 0 --quick --num-runs 10 --seed 123"

OUTPUT_DIR=${1:-"rw_eval_mgpu"}
NUM_GPUS=${2:-"1"}
JSON_FILE=""
TIMESTAMP=`date '+%Y-%m-%d_%H:%M:%S'`

NAME1[0]="dir_gs_twitter"
GRAPH[0]="market $DATA_PREFIX/${NAME1[0]}.mtx"

SUB_DIR="directed-uniform"
mkdir -p "$OUTPUT_DIR/$SUB_DIR"

TAG="variant:$SUB_DIR,num-gpus:$NUM_GPUS"

for i in {0..0}
do
   # prepare output json file name with number of gpus for this run
   JSON_FILE="${APP_NAME}__${NAME1[$i]}__GPU${NUM_GPUS}"

   #echo \
   $BIN_PREFIX$APP_NAME \
   ${GRAPH[$i]} \
   $APP_OPTIONS \
   --tag=$TAG \
   --jsonfile="$OUTPUT_DIR/$SUB_DIR/$JSON_FILE.json" \
   > "$OUTPUT_DIR/$SUB_DIR/$JSON_FILE.output.txt"
done

