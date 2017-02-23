#!/bin/bash

NVCC="$(which nvcc)"
NVCC_VERSION=$($NVCC --version | grep release | sed 's/.*release //' |  sed 's/,.*//')
GPU="k40c"

ALGO="bfs"
FIGURE="fig3"
GUNROCK_DIR="../../.."
DATADIR="$GUNROCK_DIR/dataset/large"
EXECUTION="$GUNROCK_DIR/tests/${ALGO}/bin/test_${ALGO}_${NVCC_VERSION}_x86_64"
LOGDIR="eval_$FIGURE"
BASEOPTION="--undirected --iteration-num=16 --tag=ipdps17_$FIGURE --jsondir=$LOGDIR --idempotence --src=largestdegree"
BASEFLAG=".undir.32bitSizeT.skip_pred.idempotence"

OPTION[0]="--traversal-mode=LB --queue-sizing=0 --in-sizing=0"
FLAG[0]=".LB.just-enough"

OPTION[1]="--traversal-mode=LB"
FLAG[1]=".LB.fixed"

OPTION[2]="--traversal-mode=LB --in-sizing=4"
FLAG[2]=".LB.max"

OPTION[3]="--traversal-mode=LB_CULL --queue-sizing=1.2 --in-sizing=1"
FLAG[3]=".LB_CULL.prealloc"

NAME[0]="kron_g500-logn21" && DO_A[0]="0.00001" && DO_B[0]="0.1"
NAME[1]="soc-orkut"        && DO_A[1]="0.012"  && DO_B[1]="0.1"
NAME[2]="uk-2002"          && DO_A[2]="16"     && DO_B[2]="100"

Q[0]="75" && Q_IN[0]="1"    && MAX_Q[0]="90"
Q[1]="37" && Q_IN[1]="0.4"  && MAX_Q[1]="72"
Q[2]="13" && Q_IN[2]="0.5"  && MAX_Q[2]="29"

for i in {0..2}
do
    GRAPH[$i]="market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx"
done

mkdir -p $LOGDIR

for k in {0..3}; do for i in {0..2}; do for d in 4; do
    DEVICE="0"
    for j in {1..8}
    do
        if [ "$j" -lt "$d" ]; then
            DEVICE=$DEVICE",$j"
        fi
    done

    Comm="${EXECUTION} ${GRAPH[$i]} $BASEOPTION ${OPTION[$k]} --device=$DEVICE"
    if [ "$k" -eq "1" ]; then
        Comm="$Comm --queue-sizing=${Q[$i]} --in-sizing=${Q_IN[$i]}"
    fi
    if [ "$k" -eq "2" ]; then
        Comm="$Comm --queue-sizing=${MAX_Q[$i]}"
    fi
    LogFile="$LOGDIR/${ALGO}.${GPU}x$d.${NAME[$i]}${BASEFLAG}${FLAG[$k]}.txt"

    echo ${Comm} "> $LogFile 2>&1"
         ${Comm}  > $LogFile 2>&1
    sleep 1
done;done;done
