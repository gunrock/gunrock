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
DEVICE=1

mkdir JSON

# addresses
PATH[0]="/data/clustering_dataset/KDDCUP04Bio"
PATH[1]="/data/clustering_dataset/3DRN"
PATH[2]="/home/agnes/clustering_dataset/Font/dim11"
PATH[3]="/home/agnes/clustering_dataset/Halo/dim9"
PATH[4]="/home/agnes/clustering_dataset/Halo/dim9"
PATH[5]="/home/agnes/clustering_dataset/Font/dim11"
PATH[6]="/home/agnes/clustering_dataset/Berton/dim3"
PATH[7]="/home/agnes/clustering_dataset/Berton/dim3"
PATH[8]="/home/agnes/clustering_dataset/Bower/dim3"
PATH[9]="/home/agnes/clustering_dataset/FOF"
PATH[10]="/home/agnes/clustering_dataset/FOF"
PATH[11]="/home/agnes/clustering_dataset/FOF"
PATH[12]="/home/agnes/clustering_dataset/FOF"
PATH[13]="/home/agnes/clustering_dataset/FOF"
PATH[14]="/home/agnes/clustering_dataset/FOF"
PATH[15]="/home/agnes/clustering_dataset/FOF"

# datasets
NAME[0]="KDDCUP04Bio.txt"           && K[0]="30"     && EPS[0]="5"   && MINPTS[0]="5"
NAME[1]="3D_spatial_network.txt"    && K[1]="30"     && EPS[1]="12"  && MINPTS[1]="15"
NAME[2]="DGF572K11d" 		        && K[2]="30"     && EPS[2]="12"  && MINPTS[2]="15"
NAME[3]="MPAH1M9d" 	                && K[3]="40"     && EPS[3]="15"  && MINPTS[3]="20"
NAME[4]="MPAH1.5M9d" 	            && K[4]="40"     && EPS[4]="15"  && MINPTS[4]="20"
NAME[5]="DGF2M11d" 		            && K[5]="50"     && EPS[5]="20"  && MINPTS[5]="25"
NAME[6]="MPAGB23K3d_0" 		        && K[6]="30"     && EPS[6]="12"  && MINPTS[6]="15"
NAME[7]="MPAGB8M3d_0" 		        && K[7]="50"     && EPS[7]="20"  && MINPTS[7]="25"
NAME[8]="DGB8M3d_0" 		        && K[8]="50"     && EPS[8]="20"  && MINPTS[8]="25"
NAME[9]="FOF11M3d" 		            && K[9]="30"     && EPS[9]="12"  && MINPTS[9]="15"
NAME[10]="FOF20M3d" 		        && K[10]="30"    && EPS[10]="12" && MINPTS[10]="15"
NAME[11]="FOF22M3d" 		        && K[11]="30"    && EPS[11]="12" && MINPTS[11]="15"
NAME[12]="FOF25M3d" 		        && K[12]="30"    && EPS[12]="12" && MINPTS[12]="15"
NAME[13]="FOF30M3d" 		        && K[13]="30"    && EPS[13]="12" && MINPTS[13]="15"
NAME[14]="FOF57M3d" 		        && K[14]="30"    && EPS[14]="12" && MINPTS[14]="15"
NAME[15]="FOF113M3d" 		        && K[15]="30"    && EPS[15]="12" && MINPTS[15]="15"

# shared memory
SHMEM[0]="true"     && TRANSPOSE[0]="false"
SHMEM[1]="false"    && TRANSPOSE[1]="true"

# number of threads
TH[0]="32"
TH[1]="64"
TH[2]="128"
TH[3]="256"
TH[4]="512"

# common parameters
OPTIONS="--quick=true --device=$DEVICE --NUM-THREADS=128 --tag=GUNROCK_SNN_KNN --knn-version=gunrock"

for i in {0..15}; do
    for s in {0..1}; do
        for t in {0..1}; do
            mkdir -p ./JSON/${NAME[$i]}
            DATASET="${PATH[$i]}/${NAME[$i]}"

            echo $EXECUTION market --labels-file $DATASET --k=${K[$i]} --eps=${EPS[$i]} --min-pts=${MINPTS[$i]} $OPTIONS --transpose=${TRANSPOSE[$t]} --use-shared-mem=${SHMEM[$s]} --jsondir=./JSON/${NAME[$i]}/ "> ./JSON/${NAME[$i]}.txt"
            
            $EXECUTION market --labels-file $DATASET --k=${K[$i]} --eps=${EPS[$i]} --min-pts=${MINPTS[$i]} $OPTIONS --transpose=${TRANSPOSE[$t]} --use-shared-mem=${SHMEM[$s]} --jsondir=./JSON/${NAME[$i]}/ > ./JSON/${NAME[$i]}_console.txt
        done
    done
done
