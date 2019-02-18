#!/bin/bash

OPTION[0]="--max-iter=100 --tol=0.000001"

MARK[0]=""

#get all execution files in ./bin
files=(./bin/*)
#split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
exe_file=${arr[0]}
#iterate over all file names to get the largest version number
for x in $arr
do
    output=$(grep -o "[0-9]\.[0-9]" <<<"$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

#put OS and Device type here
SUFFIX="ubuntu16.04_GTX1060"
EXECUTION=$exe_file
DATADIR="../../dataset/small"

mkdir -p eval/$SUFFIX

for i in test_hits test_pr chesapeake
do
    for j in 0
    do
        echo $EXECUTION market $DATADIR/$i.mtx ${OPTION[$j]} "> eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt"
             $EXECUTION market $DATADIR/$i.mtx ${OPTION[$j]}  > eval/$SUFFIX/$i.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done
