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

DEVICE=0
OUTPUTDIR="eval"
for i in "$@"
do
	case $i in
	--device=*)
		DEVICE="${1#*=}"
		shift
		;;
	--output-dir=*)
		OUTPUTDIR="${1#*=}"
		shift
		;;
		*)
		;;
esac
done

DEFAULT="--device=${DEVICE} --quick=true"

OPTION[0]=${DEFAULT}" --source=0 --sink=3"
OPTION[1]=${DEFAULT}" --source=3 --sink=0"
OPTION[2]=${DEFAULT}" --source=226 --sink=227"
OPTION[3]=${DEFAULT}" --source=235 --sink=236"
OPTION[4]=${DEFAULT}" --source=0 --sink=1"
OPTION[5]=${DEFAULT}" --source=3242 --sink=5435"
OPTION[6]=${DEFAULT}" --source=8922 --sink=8923"
OPTION[7]=${DEFAULT}" --source=35688 --sink=35689"

OPTION[8]=${DEFAULT}" --source=213360 --sink=213361"


EXCUTION=$exe_file
DATADIR="./_data"
LARGEDIR="/data/Taxi_Datasets/gunrock"

# Small graphs
NAME[0]="dowolny_graf2"
NAME[1]="dowolny_graf2"
NAME[2]="wei_problem_10_22"
NAME[3]="std_added"
NAME[4]="Wei_test_positive"

# Medium graphs
NAME[5]="larger"
NAME[6]="file0"
NAME[7]="file0"

SUBDIR[5]=""
SUBDIR[6]="small_run1"
SUBDIR[7]="larger"

# Large graphs
NAME[8]="file" #0 -> 13

for k in {0..4}
do
    SUFFIX="bowser_ubuntu18.04_GV100"
    mkdir -p $OUTPUTDIR/$SUFFIX

    echo $EXCUTION market $DATADIR/${NAME[$k]}.mtx ${OPTION[$k]} "> $OUTPUTDIR/$SUFFIX/${NAME[$k]}_$SUFFIX.txt"
    $EXCUTION market $DATADIR/${NAME[$k]}.mtx ${OPTION[$k]} > $OUTPUTDIR/$SUFFIX/${NAME[$k]}_$SUFFIX.txt
    sleep 1
done

for k in {5..7}
do
    SUFFIX="bowser_ubuntu18.04_GV100"
    mkdir -p $OUTPUTDIR/$SUFFIX

    echo $EXCUTION market $LARGEDIR/${SUBDIR[$k]}/${NAME[$k]}.mtx ${OPTION[$k]} "> $OUTPUTDIR/$SUFFIX/${SUBDIR[$k]}_${NAME[$k]}_$SUFFIX.txt"
	$EXCUTION market $LARGEDIR/${SUBDIR[$k]}/${NAME[$k]}.mtx ${OPTION[$k]} > $OUTPUTDIR/$SUFFIX/${SUBDIR[$k]}_${NAME[$k]}_$SUFFIX.txt
    sleep 1
done

for k in {0..13}
do
    SUFFIX="bowser_ubuntu18.04_GV100"
    mkdir -p $OUTPUTDIR/$SUFFIX

    echo $EXCUTION market $LARGEDIR/largest/${NAME[8]}${k}.mtx ${OPTION[8]} "> $OUTPUTDIR/$SUFFIX/${NAME[8]}${k}_$SUFFIX.txt"
	$EXCUTION market $LARGEDIR/largest/${NAME[8]}${k}.mtx ${OPTION[8]} > $OUTPUTDIR/$SUFFIX/${NAME[8]}${k}_$SUFFIX.txt
    sleep 1
done
