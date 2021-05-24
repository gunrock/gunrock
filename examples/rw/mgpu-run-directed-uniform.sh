# random walk: directed, uniform

#PARTITION_NAME="dgx2"
#NODE_NAME="rl-dgx2-c24-u16"

PARTITION_NAME="dgxa100_1tb"
NODE_NAME="rl-dgxa-d22-u30"

#NUM_GPUS=16
NUM_GPUS=8

APP_NAME="rw"
BIN_PREFIX="../../build/bin/"
DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/hive_datasets/mario-2TB/graphsearch"
DATA1="dir_gs_twitter" #.mtx
#DATA2="gs_twitter.values"
WALK_MODE="0" #0, 1, or 2
APP_OPTIONS="--walk-mode ${WALK_MODE} --walk-length 128 --undirected=false --store-walks 0 --quick --num-runs 10 --seed 123"

OUTPUT_DIR="eval_mgpu/$PARTITION_NAME/$NODE_NAME/directed-uniform"
mkdir -p $OUTPUT_DIR

for (( i=1; i<=$NUM_GPUS; i++))
do
    SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -w $NODE_NAME"
     $SLURM_CMD $BIN_PREFIX$APP_NAME \
                market $DATA_PREFIX/$DATA1.mtx \
                $APP_OPTIONS \
                > ./$OUTPUT_DIR/${APP_NAME}_${DATA1}_GPU$i.txt &
done

