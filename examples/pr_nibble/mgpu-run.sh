PARTITION_NAME="dgx2"
NODE_NAME="rl-dgx2-c24-u16"

#PARTITION_NAME="dgxa100_1tb"
#NODE_NAME="rl-dgxa-d22-u30"

NUM_GPUS=16
#NUM_GPUS=8

APP_NAME="pr_nibble"
BIN_PREFIX="../../build/bin/"
DATA_PREFIX="/home/u00u7u37rw7AjJoA4e357/data/gunrock/gunrock_dataset/mario-2TB/large/delaunay_n13/"
DATA1="delaunay_n13.mtx"
APP_OPTIONS="--src 0 --max-iter 1"

OUTPUT_DIR="eval_mgpu/$PARTITION_NAME/$NODE_NAME"
mkdir -p $OUTPUT_DIR

for (( i=1; i<=$NUM_GPUS; i++))
do
    SLURM_CMD="srun --cpus-per-gpu 1 -G $i -p $PARTITION_NAME -w $NODE_NAME"
    echo $SLURM_CMD $BIN_PREFIX$APP_NAME \
                market $DATA_PREFIX/$DATA1 \
                $APP_OPTIONS \
                "> ./$OUTPUT_DIR/${APP_NAME}_GPU$i.txt" # &
done


# sdp, use a bunch of data sets in the future
# for dataset in delaunay_n13 road_usa
# do
#     echo $exe_file market ../../dataset/large/$i/$i.mtx $OPTION"> ./eval/$SUFFIX/${i}.$SUFFIX${MARKS}.txt"
#          #$exe_file market ../../dataset/large/$i/$i.mtx $OPTION > ./eval/$SUFFIX/${i}.$SUFFIX${MARKS}.txt
#     #sleep 1
# done