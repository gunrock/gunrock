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

OPTION[0]=""
OPTION[0]=${OPTION[0]}" --undirected" #undirected
OPTION[0]=${OPTION[0]}" --num-runs=10" #10 runs
OPTION[0]=${OPTION[0]}" --validation=each" #validate all runs

OPTION[0]="$OPTION[0] --jsondir=./eval/$SUFFIX"


#put OS and Device type here
EXCUTION=$exe_file
DATADIR="/data/gunrock_dataset/large"

NAME[ 0]="ak2010"
NAME[ 1]="delaunay_n10"
NAME[ 2]="delaunay_n11"
NAME[ 3]="delaunay_n12"
NAME[ 4]="delaunay_n13"
NAME[ 5]="delaunay_n14"
NAME[ 6]="delaunay_n15"
NAME[ 7]="delaunay_n16"
NAME[ 8]="delaunay_n17"
NAME[ 9]="delaunay_n18"
NAME[10]="delaunay_n19"
NAME[11]="delaunay_n20"
NAME[12]="delaunay_n21"
NAME[13]="delaunay_n22"
NAME[14]="delaunay_n23"
NAME[15]="delaunay_n24"

NAME[16]="kron_g500-logn16"
NAME[17]="kron_g500-logn17"
NAME[18]="kron_g500-logn18"
NAME[19]="kron_g500-logn19"
NAME[20]="kron_g500-logn20"
NAME[21]="kron_g500-logn21"

NAME[22]="coAuthorsDBLP"
NAME[23]="coAuthorsCiteseer"
NAME[24]="coPapersDBLP"
NAME[25]="coPapersCiteseer"
NAME[26]="citationCiteseer"
NAME[27]="preferentialAttachment"
NAME[28]="soc-LiveJournal1"

NAME[29]="roadNet-CA"        
NAME[30]="belgium_osm"       
NAME[31]="netherlands_osm"   
NAME[32]="italy_osm"         
NAME[33]="luxembourg_osm"    
NAME[34]="great-britain_osm" 
NAME[35]="germany_osm"       
NAME[36]="asia_osm"          
NAME[37]="europe_osm"        
NAME[38]="road_usa"          
NAME[39]="road_central"      

NAME[40]="webbase-1M"      
NAME[41]="tweets"          
NAME[42]="bitcoin"         
NAME[43]="caidaRouterLevel"


LF[0]="0.1"
LF[1]="0.2"
LF[2]="0.3"
LF[3]="0.4"
LF[4]="0.5"
LF[5]="0.6"
LF[6]="0.7"
LF[7]="0.8"
LF[8]="0.9"
LF[9]="1.0"

#put OS and Device type here
SUFFIX="GUNROCK_vx-x-x"
mkdir -p eval/$SUFFIX

for dataset in  {22..22} 
do
    for loadfactor in {0..9}
    do
    echo $EXCUTION market $DATADIR/${NAME[$dataset]}/${NAME[$dataset]}.mtx ${OPTION[0]} --load-factor=${LF[$loadfactor]} "> eval/$SUFFIX/${NAME[$dataset]}_{$SUFFIX}_load_factor=${LF[$loadfactor]}.txt"
    $EXCUTION market $DATADIR/${NAME[$dataset]}/${NAME[$dataset]}.mtx ${OPTION[0]} --load-factor=${LF[$loadfactor]}  > eval/$SUFFIX/${NAME[$dataset]}_{$SUFFIX}_load_factor=${LF[$loadfactor]}.txt
    sleep 1
    done
done

