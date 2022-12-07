DATASET_DIR="$HOME/suitesparse"
DATASET_FILES_NAMES="$HOME"
BINARY="../build/bin"
EXE="bfs"
RESULT_DIR="./per_iter"


while read -r DATA
do
	for i in {1..10}
	do
		DATASET="$(basename -s .mtx $DATA)"
		mkdir -p $RESULT_DIR/$DATASET
		echo "$BINARY/$EXE -m $DATASET_DIR/$DATA --collect_metrics --json_dir=$RESULT_DIR/$DATASET --json_file=${DATASET}_${i}.json --num_runs=1"
		timeout 200 $BINARY/$EXE -m $DATASET_DIR/$DATA --collect_metrics --json_dir=$RESULT_DIR/$DATASET --json_file=${DATASET}_${i}.json --num_runs=1 >> $RESULT_DIR/$DATASET/${DATASET}_${i}.csv 
	done
done < "$DATASET_FILES_NAMES"
