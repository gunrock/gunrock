#!/bin/bash

NVCC="$(which nvcc)"
NVCC_VERSION=$($NVCC --version | grep release | sed 's/.*release //' |  sed 's/,.*//')
GPU="k40c"

FIGURE="fig4"
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

#NAME[0]="kron_g500-logn21" && DO_A[0]="0.00001" && DO_B[0]="0.1"

NAME[ 0]="soc-twitter-2010" && DO_A[ 0]="0.005"   && DO_B[ 0]="0.1"
NAME[ 1]="hollywood-2009"   && DO_A[ 1]="0.006"   && DO_B[ 1]="0.1"
#0.1, 0.1 for single GPU
NAME[ 2]="soc-sinaweibo"    && DO_A[ 2]="0.00005" && DO_B[ 2]="0.1"
NAME[ 3]="soc-orkut"        && DO_A[ 3]="0.012"   && DO_B[ 3]="0.1"
NAME[ 4]="soc-LiveJournal1" && DO_A[ 4]="0.200"   && DO_B[ 4]="0.1"
NAME[ 5]="uk-2002"          && DO_A[ 5]="16"      && DO_B[ 5]="100"
NAME[ 6]="uk-2005"          && DO_A[ 6]="10"      && DO_B[ 6]="20"
NAME[ 7]="webbase-2001"     && DO_A[ 7]="100"     && DO_B[ 7]="100"
#2, 1.2 for memory
NAME[ 8]="indochina-2004"   && DO_A[ 8]="100"     && DO_B[ 8]="100"
NAME[ 9]="arabic-2005"      && DO_A[ 9]="100"     && DO_B[ 9]="100"
NAME[10]="rmat_n20_512"     && DO_A[10]="0.00001" && DO_B[10]="0.1" && SCALE[10]="20" && EF[10]="512"
NAME[11]="rmat_n21_256"     && DO_A[11]="0.00001" && DO_B[11]="0.1" && SCALE[11]="21" && EF[11]="256"
NAME[12]="rmat_n22_128"     && DO_A[12]="0.00001" && DO_B[12]="0.1" && SCALE[12]="22" && EF[12]="128"
NAME[13]="rmat_n23_64"      && DO_A[13]="0.00001" && DO_B[13]="0.1" && SCALE[13]="23" && EF[13]="64"
NAME[14]="rmat_n24_32"      && DO_A[14]="0.00001" && DO_B[14]="0.1" && SCALE[14]="24" && EF[14]="32"
NAME[15]="rmat_n25_16"      && DO_A[15]="0.00001" && DO_B[15]="0.1" && SCALE[15]="25" && EF[15]="16"
NAME[16]="europe_osm"       && DO_A[16]="2.5"     && DO_B[16]="15"
NAME[17]="asia_osm"         && DO_A[17]="1.5"     && DO_B[17]="10"
NAME[18]="germany_osm"      && DO_A[18]="1.5"     && DO_B[18]="10"
NAME[19]="road_usa"         && DO_A[19]="1.0"     && DO_B[19]="10"
NAME[20]="road_central"     && DO_A[20]="1.2"     && DO_B[20]="10"

mkdir -p $LOGDIR

for i in {0..20}; do for k in {0..6}; do for d in {1..4}; do
    DEVICE="0"
    for j in {1..8}
    do
        if [ "$j" -lt "$d" ]; then
            DEVICE=$DEVICE",$j"
        fi
    done

    if [[ "$i" -le "9" || "$i" -ge "16" ]]; then
        GRAPH="market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx"
    else
        if [ "$d" -eq "1" ]; then
            GRAPH="rmat --rmat_scale=${SCALE[$i]} --rmat_edgefactor=${EF[$i]}"
        else
            GRAPH="grmat --rmat_scale=${SCALE[$i]} --rmat_edgefactor=${EF[$i]}"
        fi
    fi

    Comm="${EXECUTION[$k]} ${GRAPH} $BASEOPTION ${OPTION[$k]} --device=$DEVICE"
    if [ "$k" -le "2" ]; then
        if [ "$i" -eq "7" ]; then
            Comm="$Comm --queue-sizing=2 --in-sizing=1.2"
        else
            Comm="$Comm --queue-sizing=6.5 --in-sizing=4"
        fi
    fi
    if [[ "$k" -eq "0" || "$k" -eq "2" ]]; then
        if [[ "$i" -eq "1" && "$d" -eq "1" ]]; then
            Comm="$Comm --do_a=0.1 --do_b=0.1"
        else
            Comm="$Comm --do_a=${DO_A[$i]} --do_b=${DO_B[$i]}"
        fi
    fi

    LogFile="$LOGDIR/${ALGO[$k]}.${GPU}x$d.${NAME[$i]}${BASEFLAG}${FLAG[$k]}.txt"

    echo ${Comm} "> $LogFile 2>&1"
         ${Comm}  > $LogFile 2>&1
    sleep 1
done;done;done
