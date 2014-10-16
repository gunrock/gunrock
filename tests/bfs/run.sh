#!/bin/bash

OPTION[0]="--src=largestdegree --device=0,1,2,3 --partition_method=random --grid-size=768"
#OPTION[0]="" #directed and do not mark-pred"
OPTION[1]=${OPTION[0]}" --mark-pred" #directed and mark-pred"
OPTION[2]=${OPTION[0]}" --undirected" #undirected and do not mark-pred"
OPTION[3]=${OPTION[1]}" --undirected" #undirected and mark-pred"
OPTION[4]=${OPTION[0]}" --idempotence"
OPTION[5]=${OPTION[1]}" --idempotence"
OPTION[6]=${OPTION[2]}" --idempotence"
OPTION[7]=${OPTION[3]}" --idempotence"

MARK[0]=""
MARK[1]=${MARK[0]}".mark_pred"
MARK[2]=${MARK[0]}".undir"
MARK[3]=${MARK[1]}".undir"
MARK[4]=${MARK[0]}".idempotence"
MARK[5]=${MARK[1]}".idempotence"
MARK[6]=${MARK[2]}".idempotence"
MARK[7]=${MARK[3]}".idempotence"

#put OS and Device type here
SUFFIX="ubuntu12.04.k40cx4_rand"
EXCUTION="./bin/test_bfs_6.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

mkdir -p eval/$SUFFIX

NAME[0]="ak2010"       && SIZE[0]="16"
NAME[1]="delaunay_n10" && SIZE[1]="16"
NAME[2]="delaunay_n11" && SIZE[2]="16"
NAME[3]="delaunay_n12" && SIZE[3]="16"
NAME[4]="delaunay_n13" && SIZE[4]="16"
NAME[5]="delaunay_n14" && SIZE[5]="16"
NAME[6]="delaunay_n15" && SIZE[6]="16"
NAME[7]="delaunay_n16" && SIZE[7]="16"
NAME[8]="delaunay_n17" && SIZE[8]="16"
NAME[9]="delaunay_n18" && SIZE[9]="16"
NAME[10]="delaunay_n19" && SIZE[10]="16"
NAME[11]="delaunay_n20" && SIZE[11]="16"
NAME[12]="delaunay_n21" && SIZE[12]="16"
NAME[13]="delaunay_n22" && SIZE[13]="16"
NAME[14]="delaunay_n23" && SIZE[14]="16"
NAME[15]="delaunay_n24" && SIZE[15]="16"
NAME[16]="kron_g500-logn16" && SIZE[16]="256" 
NAME[17]="kron_g500-logn17" && SIZE[17]="256" 
NAME[18]="kron_g500-logn18" && SIZE[18]="256" 
NAME[19]="kron_g500-logn19" && SIZE[19]="256" 
NAME[20]="kron_g500-logn20" && SIZE[20]="128" 
NAME[21]="kron_g500-logn21" && SIZE[21]="128" 
NAME[22]="coAuthorsDBLP"    && SIZE[22]="32"
NAME[23]="soc-LiveJournal1" && SIZE[23]="16"
NAME[24]="webbase-1M"       && SIZE[24]="32"
NAME[25]="tweets"           && SIZE[25]="32"
NAME[26]="bitcoin"          && SIZE[26]="32"
NAME[27]="roadnet"          && SIZE[27]="32"
NAME[28]="road_usa"         && SIZE[28]="32"
NAME[29]="road_central"     && SIZE[29]="32"
NAME[30]="belgium_osm"      && SIZE[30]="32"
NAME[31]="netherlands_osm"  && SIZE[31]="32"
NAME[32]="italy_osm"        && SIZE[32]="32"
NAME[33]="luxembourg_osm"   && SIZE[33]="32"
NAME[34]="great-britain_osm" && SIZE[34]="32"
NAME[35]="germany_osm"      && SIZE[35]="32"
NAME[36]="asia_osm"         && SIZE[36]="32"
NAME[37]="europe_osm"       && SIZE[37]="16"

for i in {0..37} 
do
    for j in 0 1 2 3 4 6
    do
        echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${SIZE[$i]}"> eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARK[$j]}.txt"
        $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${SIZE[$i]} > eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARK[$j]}.txt
        sleep 1
    done
done
