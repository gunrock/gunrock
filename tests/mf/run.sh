#!/bin/bash

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

DEFAULT="--device=2 --quick=true"

OPTION[0]=${DEFAULT}" --source=0 --sink=3"
OPTION[1]=${DEFAULT}" --source=3 --sink=0"
OPTION[2]=${DEFAULT}" --source=226 --sink=227"
OPTION[3]=${DEFAULT}" --source=235 --sink=236"
OPTION[4]=${DEFAULT}" --source=0 --sink=1"
OPTION[5]=${DEFAULT}" --source=3242 --sink=5435"
OPTION[6]=${DEFAULT}" --source=8922 --sink=8923"
OPTION[7]=${DEFAULT}" --source=35688 --sink=35689"




EXCUTION=$exe_file
DATADIR="./_data"
LARGEDIR="/data/Taxi_Datasets/gunrock"

NAME[0]="$DATADIR/dowolny_graf2"
NAME[1]="$DATADIR/dowolny_graf2"
NAME[2]="$DATADIR/wei_problem_10_22"
NAME[3]="$DATADIR/std_added.mtx"
NAME[4]="$DATADIR/Wei_test_positive"
NAME[5]="$LARGEDIR/larger.mtx"
NAME[6]="$LARGEDIR/small_run1/file0.mtx"
NAME[7]="$LARGEDIR/larger/file0.mtx"
 
for k in {0..7}
do
    SUFFIX="ubuntu16.04_TitanV"
    mkdir -p eval/$SUFFIX

    echo $EXCUTION market ${NAME[$k]} ${OPTION[$k]} "> eval/$SUFFIX/${k}_$SUFFIX.txt"
    $EXCUTION market ${NAME[$k]} ${OPTION[$k]} > eval/$SUFFIX/${k}_$SUFFIX.txt
    sleep 1
done

