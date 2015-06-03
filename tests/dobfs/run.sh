#!/bin/sh

OPTION1="" #directed and do not mark-pred"
OPTION2="--mark-pred" #directed and mark-pred"
OPTION3="--undirected" #undirected and do not mark-pred"
OPTION4="--undirected --mark-pred" #undirected and mark-pred"

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
SUFFIX="ubuntu12.04.k40c"

mkdir -p eval/$SUFFIX

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 soc-LiveJournal1 kron_g500-logn21 webbase-1M
do
    echo $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION1
         $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION1 > eval/$SUFFIX/$i.$SUFFIX.dir_no_mark_pred.txt
    sleep 1
    echo $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION2
         $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION2 > eval/$SUFFIX/$i.$SUFFIX.dir_mark_pred.txt
    sleep 1
    echo $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION3
         $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION3 > eval/$SUFFIX/$i.$SUFFIX.undir_no_mark_pred.txt
    sleep 1
    echo $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION4
         $exe_file market ../../dataset/large/$i/$i.mtx --src=largestdegree $OPTION4 > eval/$SUFFIX/$i.$SUFFIX.undir_mark_pred.txt
    sleep 1
done
