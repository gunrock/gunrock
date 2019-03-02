#!/bin/bash

OPTION=""

# --quick running without CPU reference algorithm, if you want to test CPU
# reference algorithm, delete $OPTION2 in some lines. Warning: for large
# data this can take a long time.

# get all execution files in ./bin
files=./bin/*

# split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
exe_file=""

# iterate over all file names to get the largest version number
for x in $arr;
do
    output=$(grep -o "[0-9]\.[0-9]" <<< "$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

# put OS and Device here
SUFFIX="GUNROCK_v0-5-0"

mkdir -p eval/$SUFFIX

for i in belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21
do
    echo $exe_file market ../../dataset/large/$i/$i.mtx $OPTION
         $exe_file market ../../dataset/large/$i/$i.mtx $OPTION > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 1
done
