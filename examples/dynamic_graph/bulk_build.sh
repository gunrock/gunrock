#!/bin/bash

#get all execution files in ./bin
files=./bin/*
#split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"

exe_file=""
#iterate over all file names to get the largest version number
for x in $arr
do
    output=$(grep -o "[0-9]\.[0-9]" <<< "$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

OPTION[0]=""
OPTION[0]=${OPTION[0]}" --undirected" #undirected
OPTION[0]=${OPTION[0]}" --num-runs=10" #10 runs
OPTION[0]=${OPTION[0]}" --validation=each" #validate all runs

OPTION[0]=${OPTION[0]}" --jsondir=./eval/$SUFFIX"


#put OS and Device type here
EXCUTION=$exe_file

DATADIR="../../dataset/large/"

test_datasets=()
ALL_IN_DIR="${DATADIR}*"
for f in $ALL_IN_DIR
do 
    if [ -d "$f" ]; then 
        dir_name=$(basename "${f}")
        test_datasets=(${test_datasets[*]} "$dir_name")
    fi
done

LF[0]="0.2"
LF[1]="0.3"
LF[2]="0.4"
LF[3]="0.5"
LF[4]="0.6"
LF[5]="0.7"
LF[6]="0.8"
LF[7]="0.9"
LF[8]="1.0"

#put OS and Device type here
SUFFIX="GUNROCK_vx-x-x"
mkdir -p eval/$SUFFIX

for dataset in  "${test_datasets[@]}"
do
    for loadfactor in {0..7}
    do
    echo $EXCUTION market $DATADIR$dataset/$dataset.mtx ${OPTION[0]} --load-factor=${LF[$loadfactor]} "> eval/$SUFFIX/{$dataset}_{$SUFFIX}_load_factor=${LF[$loadfactor]}.txt"
    $EXCUTION market $DATADIR/$dataset/$dataset.mtx ${OPTION[0]} --load-factor=${LF[$loadfactor]}  > eval/$SUFFIX/$dataset"_"$SUFFIX"_load_factor"=${LF[$loadfactor]}.txt
    sleep 1
    done
done

