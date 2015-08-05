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

mkdir -p eval/PPOPP15
for i in  1-soc 2-bitcoin 3-kron 6-roadnet
do
    echo $exe_file market /data/PPOPP15/$i.mtx --src=0 --iteration-num=10
         $exe_file market /data/PPOPP15/$i.mtx --src=0 --iteration-num=10  > eval/PPOPP15/$i.txt
    sleep 1
done
