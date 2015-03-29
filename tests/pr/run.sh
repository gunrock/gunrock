#!/bin/sh

OPTION1="--undirected" #undirected
OPTION2="--quick" #quick running without CPU reference algorithm, if you want to test CPU reference algorithm, delete $OPTION2 in some lines. Warning: for large data this can take a long time.

#get all execution files in ./bin
files=$(./bin/*)
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

for i in ak2010 belgium_osm coAuthorsDBLP delaunay_n13 delaunay_n21 
do
    echo $exe_file market ../../dataset/large/$i/$i.mtx $OPTION1
         $exe_file market ../../dataset/large/$i/$i.mtx $OPTION1 > eval/$SUFFIX/$i.$SUFFIX.undir.txt
    sleep 1
done

for i in soc-LiveJournal1 kron_g500-logn21
do
    echo $exe_file market ../../dataset/large/$i/$i.mtx $OPTION1 $OPTION2
         $exe_file market ../../dataset/large/$i/$i.mtx $OPTION1 $OPTION2 > eval/$SUFFIX/$i.$SUFFIX.undir_quick.txt
    sleep 1
done

echo $exe_file market ../../dataset/large/webbase-1M/webbase-1M.mtx $OPTION2
     $exe_file market ../../dataset/large/webbase-1M/webbase-1M.mtx $OPTION2 > eval/$SUFFIX/webbase-1M.$SUFFIX.quick.txt

