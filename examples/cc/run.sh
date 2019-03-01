#!/bin/bash

#get all execution files in ./bin
files=./bin/*

#split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
exe_file=""
#iterate over all file names to get the largest version number
for x in $arr;
do
    output=$(grep -o "[0-9]\.[0-9]" <<< "$x")
    if [ "$output" \> "$max_ver_num" ]; then
        exe_file=$x
    fi
done

#put OS and Device type here
SUFFIX="GUNROCK_v0-5-0"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M;
do
    echo $exe_file market ../../dataset/large/$i/$i.mtx
         $exe_file market ../../dataset/large/$i/$i.mtx > eval/$SUFFIX/$i.$SUFFIX.txt
    sleep 1
done


