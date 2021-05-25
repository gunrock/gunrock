#!/bin/bash

PARTITION_NAME="dgx2"
NUM_GPUS=2

#APP_SCRIPT="./hive-rw-test.sh"
APP_SCRIPT[0]="./hive-rw-undirected-uniform.sh"
APP_SCRIPT[1]="./hive-rw-directed-uniform.sh"

OUTPUT_DIR="rw_eval_mgpu/$PARTITION_NAME"
mkdir -p $OUTPUT_DIR

for app in {0..1}
do
   for (( i=1; i<=$NUM_GPUS; i++))
   do
       # prepare and run SLURM command
       #SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -w $NODE_NAME"

       SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME "
       $SLURM_CMD ${APP_SCRIPT[$app]} $OUTPUT_DIR $i &

       #echo "$SLURM_CMD ${APP_SCRIPT[$app]} $OUTPUT_DIR $i"
       #sleep 1
   done
done
