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

OPTION[0]="--src=largestdegree --device=0 --partition-method=biasrandom --grid-size=768"
#OPTION[0]="" #directed and do not mark-pred"
OPTION[1]=${OPTION[0]}" --undirected" #undirected and do not mark-pred"
OPTION[2]=${OPTION[0]}" --mark-path"  #directed and mark-pred"
OPTION[3]=${OPTION[1]}" --mark-path" #undirected and mark-pred"

MARK[0]=""
MARK[1]=${MARK[0]}"_undir"
MARK[2]=${MARK[0]}"_markpath"
MARK[3]=${MARK[1]}"_markpath"

#put OS and Device type here
EXCUTION=$exe_file
DATADIR="../../dataset/large"

NAME[ 0]="delaunay_n13"      && Q_SIZE_DIR[ 4]="0.25" && I_SIZE_DIR[ 4]="0.15" && Q_SIZE_UDIR[ 4]="1.40" && I_SIZE_UDIR[ 4]="0.30"

F[0]="0.0" && F[1]="0.1" && F[2]="0.2" && F[3]="0.3" && F[4]="0.4" && F[5]="0.5" && F[6]="0.6" && F[7]="0.7" && F[8]="0.8" && F[9]="0.9"
F[10]="1.0"
 
for k in {0..0}
do
    #put OS and Device type here
    SUFFIX="ubuntu16.04_TitanVx2_brp${F[$k]}"
    mkdir -p eval/$SUFFIX

    for i in  {0..0} 
    do
        for j in {0..0}
        do
            if [ "$j" -eq "0" ] || [ "$j" -eq "2" ] ; then
                echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} --traversal-mode=SIMPLE"> eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt"
                $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} --traversal-mode=SIMPLE > eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt
            else
                echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} --traversal-mode=SIMPLE "> eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt"
                $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} --traversal-mode=SIMPLE> eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt
            fi
            sleep 1
        done
    done
done

