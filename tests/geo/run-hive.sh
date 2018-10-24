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

DEFAULT="--device=0 --geo-complete=false"

OPTION[0]=${DEFAULT}" --geo-iter=3 --spatial-iter=1000"
OPTION[1]=${DEFAULT}" --geo-iter=3 --spatial-iter=1000" 
OPTION[2]=${DEFAULT}" --geo-iter=3 --spatial-iter=1000" 
OPTION[3]=${DEFAULT}" --geo-iter=3 --spatial-iter=1000 --quick --quiet"

EXCUTION=$exe_file
DATADIR="./_data/"

NAME[0]="sample"
NAME[1]="instagram-sample"
NAME[2]="instagram"
NAME[3]="twitter" # runs out of memory
 
for k in {0..2}
do
    SUFFIX="ubuntu16.04_TitanV"
    mkdir -p eval/$SUFFIX

    echo $EXCUTION market $DATADIR/${NAME[$k]}/graph/${NAME[$k]}.mtx --labels-file=$DATADIR/${NAME[$k]}/graph/${NAME[$k]}.labels ${OPTION[$k]} "> eval/$SUFFIX/${NAME[$k]}_$SUFFIX.txt"
    $EXCUTION market $DATADIR/${NAME[$k]}/graph/${NAME[$k]}.mtx --labels-file=$DATADIR/${NAME[$k]}/graph/${NAME[$k]}.labels ${OPTION[$k]} > eval/$SUFFIX/${NAME[$k]}_$SUFFIX.txt
    sleep 1
done

