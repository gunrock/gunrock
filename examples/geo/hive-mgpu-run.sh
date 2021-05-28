#!/bin/bash

PARTITION_NAME="dgx2"
#PARTITION_NAME="batch"
NODE_NAME="dgx2-000"
NUM_GPUS=${1:-"16"}

APP_SCRIPT="./hive-geo-test.sh"
GEO_ITER[0]=10
GEO_ITER[1]=100

SPATIAL_ITER[0]=1000
SPATIAL_ITER[1]=10000

OUTPUT_DIR="geo_eval_mgpu/$PARTITION_NAME"
mkdir -p $OUTPUT_DIR

for gi in {0..1} #geo_iter
do
   for si in {0..1} #spatial_iter
   do
      for (( i=1; i<=$NUM_GPUS; i++))
      do
          # prepare and run SLURM command
          #SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -w $NODE_NAME"
          #SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME "
      
          SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -N 1 "
          $SLURM_CMD $APP_SCRIPT $OUTPUT_DIR $i ${GEO_ITER[$gi]} ${SPATIAL_ITER[$si]} &
      
          #echo "$SLURM_CMD $APP_SCRIPT $OUTPUT_DIR $i ${GEO_ITER[$gi]} ${SPATIAL_ITER[$si]}"
          #sleep 1
      done
   done
done
