#!/bin/bash

NVCC="$(which nvcc)"
NVCC_VERSION=$($NVCC --version | grep release | sed 's/.*release //' |  sed 's/,.*//')
GPU="k40c"

FIGURE="fig2"
GUNROCK_DIR="../../.."
DATADIR="$GUNROCK_DIR/dataset/large"
PARTITIONERS="random biasrandom metis"
LOGDIR="eval_$FIGURE"
BASEOPTION="--undirected --iteration-num=16 --tag=ipdps17_$FIGURE --jsondir=$LOGDIR"
BASEFLAG=".undir.32bitSizeT"

ALGO[0]="bfs"
OPTION[0]="--idempotence --direction-optimized --src=largestdegree --traversal-mode=LB_CULL --queue-sizing=6.5 --in-sizing=4"
FLAG[0]=".skip_pred.idempotence.LB_CULL.do"

ALGO[1]="bfs"
OPTION[1]="--idempotence --src=largestdegree --traversal-mode=LB_CULL --queue-sizing=6.5 --in-sizing=4"
FLAG[1]=".skip_pred.idempotence.LB_CULL.fw"

ALGO[2]="pr"
OPTION[2]="--normalized --quick --traversal-mode=LB"
FLAG[2]=".normalized.nocomp"

for k in {0..2}
do
    EXECUTION[k]="$GUNROCK_DIR/tests/${ALGO[k]}/bin/test_${ALGO[k]}_${NVCC_VERSION}_x86_64"
done

NAME[0]="kron_g500-logn21" && DO_A[0]="0.00001" && DO_B[0]="0.1"
NAME[1]="soc-orkut"        && DO_A[1]="0.012"  && DO_B[1]="0.1"
NAME[2]="uk-2002"          && DO_A[2]="16"     && DO_B[2]="100"

for i in {0..2}
do
    GRAPH[$i]="market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx"
done

mkdir -p $LOGDIR

for k in {0..2}; do for partitioner in $PARTITIONERS; do for i in {0..2}; do for d in 1 4; do
    DEVICE="0"
    for j in {1..8}
    do
        if [ "$j" -lt "$d" ]; then
            DEVICE=$DEVICE",$j"
        fi
    done

    Comm="${EXECUTION[$k]} ${GRAPH[$i]} $BASEOPTION ${OPTION[$k]} --partition-method=$partitioner --device=$DEVICE --do_a=${DO_A[$i]} --do_b=${DO_B[$i]} "
    LogFile="$LOGDIR/${ALGO[$k]}.${GPU}x$d.${NAME[$i]}${BASEFLAG}${FLAG[$k]}.$partitioner.txt"
    echo ${Comm} "> $LogFile 2>&1"
         ${Comm}  > $LogFile 2>&1

done;done;done;done
