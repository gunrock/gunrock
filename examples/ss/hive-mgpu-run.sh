#!/bin/bash

PARTITION_NAME="dgx2"
NUM_GPUS=16

OUTPUT_DIR="eval_mgpu/$PARTITION_NAME"
JSON_FILE=""
TIMESTAMP=""
TAG=""

APP_SCRIPT="./hive-ss-test.sh"

mkdir -p $OUTPUT_DIR

for (( i=1; i<=$NUM_GPUS; i++))
do
    # prepare output json file name with number of gpus for this run
    TIMESTAMP=`date '+%Y-%m-%d_%H:%M:%S'`
    TAG=$TIMESTAMP
    JSON_FILE="ss__${TAG}__GPU${i}"

    # prepare and run SLURM command
    #SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -w $NODE_NAME"
    SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME "
    #echo "$SLURM_CMD $APP_SCRIPT $OUTPUT_DIR/$JSON_FILE"
    $SLURM_CMD $APP_SCRIPT $OUTPUT_DIR/$JSON_FILE &
    #sleep 1
done
