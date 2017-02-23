#!/bin/bash

NVCC="$(which nvcc)"
NVCC_VERSION=$($NVCC --version | grep release | sed 's/.*release //' |  sed 's/,.*//')
GPU="k40c"

FIGURE="fig5"
GUNROCK_DIR="../../.."
DATADIR="$GUNROCK_DIR/dataset/large"
LOGDIR="eval_$FIGURE"
BASEOPTION="--undirected --iteration-num=16 --partition-method=random --tag=ipdps17_$FIGURE --jsondir=$LOGDIR"
BASEFLAG=".undir.32bitSizeT"

ALGO[0]="bfs"
OPTION[0]="--traversal-mode=LB_CULL --idempotence --src=largestdegree --direction-optimized"
FLAG[0]=".skip_pred.LB_CULL.idempotence.do"

ALGO[1]="bfs"
OPTION[1]="--traversal-mode=LB_CULL --idempotence --src=largestdegree"
FLAG[1]=".skip_pred.LB_CULL.idempotence.fw"

ALGO[2]="bfs"
OPTION[2]="--traversal-mode=LB_CULL --src=largestdegree --direction-optimized"
FLAG[2]=".skip_pred.LB_CULL.do"

ALGO[3]="sssp"
OPTION[3]="--traversal-mode=LB --queue-sizing=1 --in-sizing=1"
FLAG[3]=".LB"

ALGO[4]="cc"
OPTION[4]="--in-sizing=1.1"
FLAG[4]=""

ALGO[5]="bc"
OPTION[5]="--src=largestdegree --traversal-mode=LB_CULL --in-sizing=1.1 --queue-sizing=1.2"
FLAG[5]=".skip_pred.LB_CULL"

ALGO[6]="pr"
OPTION[6]="--quick --normalized --traversal-mode=LB"
FLAG[6]=".nocomp.normalized"

for k in {0..6}; do
    EXECUTION[k]="$GUNROCK_DIR/tests/${ALGO[$k]}/bin/test_${ALGO[$k]}_${NVCC_VERSION}_x86_64"
done

mkdir -p $LOGDIR

for i in {0..2}; do for k in 0 1 2 6; do for d in {1..4}; do
    DEVICE="0"
    for j in {1..8}
    do
        if [ "$j" -lt "$d" ]; then
            DEVICE=$DEVICE",$j"
        fi
    done
    queue_sizing=$(echo "1.25 * ${d}" | bc)

    if [ "$d" -eq "1" ]; then
        rmat_method="rmat"
    else
        rmat_method="grmat"
    fi

    if [ "$i" -eq "0" ]; then
        GRAPH="$rmat_method --rmat_scale=24 --rmat_edgefactor=32"
        NAME="rmat_n24_32"
    elif [ "$i" -eq "1" ]; then
        num_nodes=$(echo "(2^19) * ${d}" | bc)
        if [ "$d" -eq "8" ]; then
            edge_factor="255"
        else
            edge_factor="256"
        fi
        GRAPH="$rmat_method --rmat_nodes=${num_nodes} --rmat_edgefactor=${edge_factor}"
        NAME="rmat_${num_nodes}_${edge_factor}"
    elif [ "$i" -eq "2" ]; then
        edge_factor=$(echo "256 * ${d}" | bc)
        if [ "$d" -eq "8" ]; then
            edge_factor=$(echo "${edge_factor} - 1" | bc)
        fi
        GRAPH="$rmat_method --rmat_scale=19 --rmat_edgefactor=${edge_factor}"
        NAME="rmat_n19_${edge_factor}"
    fi

    Comm="${EXECUTION[$k]} ${GRAPH} $BASEOPTION ${OPTION[$k]} --device=$DEVICE"
    if [ "$k" -le "2" ]; then
        Comm="$Comm --queue-sizing=${queue_sizing} --in-sizing=5"
    fi
    if [[ "$k" -eq "0" || "$k" -eq "2" ]]; then
        Comm="$Comm --do_a=0.00001 --do_b=0.1"
    fi

    LogFile="$LOGDIR/${ALGO[$k]}.${GPU}x$d.${NAME}${BASEFLAG}${FLAG[$k]}.txt"

    echo ${Comm} "> $LogFile 2>&1"
         ${Comm}  > $LogFile 2>&1
    sleep 1
done;done;done
