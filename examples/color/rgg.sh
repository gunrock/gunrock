#!/bin/bash

# get all execution files in ./bin
files=(./bin/*)
# split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"

exe_file=${arr[0]}
# iterate over all file names to get the largest version number
for x in $arr
do
    output=$(grep -o "[0-9]\.[0-9]" <<<"$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

EXECUTION=$exe_file

# addresses
DATADIR="/data/gunrock_dataset/large"
NAME[0]="rgg_n_2_15_s0"
NAME[1]="rgg_n_2_16_s0"
NAME[2]="rgg_n_2_17_s0"
NAME[3]="rgg_n_2_18_s0"
NAME[4]="rgg_n_2_19_s0"
NAME[5]="rgg_n_2_20_s0"
NAME[6]="rgg_n_2_21_s0"
NAME[7]="rgg_n_2_22_s0"
NAME[8]="rgg_n_2_23_s0"
NAME[9]="rgg_n_2_24_s0"

OPTIONS="--JPL=true --test-run=true --undirected --tag=Size-vs-Colors-Performance"

for i in {0..9}
do
    SUFFIX="K40C_Ubunru_16-04"
    mkdir -p eval/$SUFFIX
    DEVICE="0"

    echo $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTIONS --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/${NAME[$i]}.txt"
         $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTIONS --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/${NAME[$i]}.txt
        sleep 1
done
