#!/bin/bash

OPTION[0]="--src=largestdegree --device=0,1,2,3 --partition_method=biasrandom --grid-size=768 --disable-size-check"
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
SUFFIX="ubuntu12.04.k40cx4_brand0.5_dsize"
EXCUTION="./bin/test_bfs_6.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

mkdir -p eval/$SUFFIX

NAME[ 0]="ak2010"            && Q_SIZE_DIR[ 0]="0.05" && I_SIZE_DIR[ 0]="0.02" && Q_SIZE_UDIR[ 0]="0.25" && I_SIZE_UDIR[ 0]="0.08"

NAME[ 1]="delaunay_n10"      && Q_SIZE_DIR[ 1]="2.00" && I_SIZE_DIR[ 1]="0.40" && Q_SIZE_UDIR[ 1]="2.00" && I_SIZE_UDIR[ 1]="0.40"
NAME[ 2]="delaunay_n11"      && Q_SIZE_DIR[ 2]="2.00" && I_SIZE_DIR[ 2]="0.40" && Q_SIZE_UDIR[ 2]="2.00" && I_SIZE_UDIR[ 2]="0.40"
NAME[ 3]="delaunay_n12"      && Q_SIZE_DIR[ 3]="2.00" && I_SIZE_DIR[ 3]="0.40" && Q_SIZE_UDIR[ 3]="2.00" && I_SIZE_UDIR[ 3]="0.40"
NAME[ 4]="delaunay_n13"      && Q_SIZE_DIR[ 4]="0.25" && I_SIZE_DIR[ 4]="0.15" && Q_SIZE_UDIR[ 4]="0.25" && I_SIZE_UDIR[ 4]="0.15"
NAME[ 5]="delaunay_n14"      && Q_SIZE_DIR[ 5]="0.25" && I_SIZE_DIR[ 5]="0.15" && Q_SIZE_UDIR[ 5]="0.25" && I_SIZE_UDIR[ 5]="0.15"
NAME[ 6]="delaunay_n15"      && Q_SIZE_DIR[ 6]="0.25" && I_SIZE_DIR[ 6]="0.15" && Q_SIZE_UDIR[ 6]="0.25" && I_SIZE_UDIR[ 6]="0.15"
NAME[ 7]="delaunay_n16"      && Q_SIZE_DIR[ 7]="0.25" && I_SIZE_DIR[ 7]="0.15" && Q_SIZE_UDIR[ 7]="0.25" && I_SIZE_UDIR[ 7]="0.15"
NAME[ 8]="delaunay_n17"      && Q_SIZE_DIR[ 8]="0.25" && I_SIZE_DIR[ 8]="0.15" && Q_SIZE_UDIR[ 8]="0.25" && I_SIZE_UDIR[ 8]="0.15"
NAME[ 9]="delaunay_n18"      && Q_SIZE_DIR[ 9]="0.25" && I_SIZE_DIR[ 9]="0.15" && Q_SIZE_UDIR[ 9]="0.25" && I_SIZE_UDIR[ 9]="0.15"
NAME[10]="delaunay_n19"      && Q_SIZE_DIR[10]="0.25" && I_SIZE_DIR[10]="0.15" && Q_SIZE_UDIR[10]="0.25" && I_SIZE_UDIR[10]="0.15"
NAME[11]="delaunay_n20"      && Q_SIZE_DIR[11]="0.25" && I_SIZE_DIR[11]="0.15" && Q_SIZE_UDIR[11]="0.25" && I_SIZE_UDIR[11]="0.15"
NAME[12]="delaunay_n21"      && Q_SIZE_DIR[12]="0.25" && I_SIZE_DIR[12]="0.15" && Q_SIZE_UDIR[12]="0.25" && I_SIZE_UDIR[12]="0.15"
NAME[13]="delaunay_n22"      && Q_SIZE_DIR[13]="0.25" && I_SIZE_DIR[13]="0.15" && Q_SIZE_UDIR[13]="0.25" && I_SIZE_UDIR[13]="0.15"
NAME[14]="delaunay_n23"      && Q_SIZE_DIR[14]="0.25" && I_SIZE_DIR[14]="0.15" && Q_SIZE_UDIR[14]="0.25" && I_SIZE_UDIR[14]="0.15"
NAME[15]="delaunay_n24"      && Q_SIZE_DIR[15]="0.25" && I_SIZE_DIR[15]="0.15" && Q_SIZE_UDIR[15]="0.25" && I_SIZE_UDIR[15]="0.15"

NAME[16]="kron_g500-logn16"  && Q_SIZE_DIR[16]="75.0" && I_SIZE_DIR[16]="1.50" && Q_SIZE_UDIR[16]="150 " && I_SIZE_UDIR[16]="1.50" 
NAME[17]="kron_g500-logn17"  && Q_SIZE_DIR[17]="75.0" && I_SIZE_DIR[17]="1.50" && Q_SIZE_UDIR[17]="150 " && I_SIZE_UDIR[17]="1.50" 
NAME[18]="kron_g500-logn18"  && Q_SIZE_DIR[18]="75.0" && I_SIZE_DIR[18]="1.50" && Q_SIZE_UDIR[18]="150 " && I_SIZE_UDIR[18]="1.50" 
NAME[19]="kron_g500-logn19"  && Q_SIZE_DIR[19]="75.0" && I_SIZE_DIR[19]="1.50" && Q_SIZE_UDIR[19]="150 " && I_SIZE_UDIR[19]="1.50" 
NAME[20]="kron_g500-logn20"  && Q_SIZE_DIR[20]="75.0" && I_SIZE_DIR[20]="1.50" && Q_SIZE_UDIR[20]="150 " && I_SIZE_UDIR[20]="1.50" 
NAME[21]="kron_g500-logn21"  && Q_SIZE_DIR[21]="75.0" && I_SIZE_DIR[21]="1.50" && Q_SIZE_UDIR[21]="150 " && I_SIZE_UDIR[21]="1.50" 

NAME[22]="coAuthorsDBLP"     && Q_SIZE_DIR[22]="0.60" && I_SIZE_DIR[22]="0.33" && Q_SIZE_UDIR[22]="2.50" && I_SIZE_UDIR[22]="0.50" 
NAME[23]="coAuthorsCiteseer" && Q_SIZE_DIR[23]="0.60" && I_SIZE_DIR[23]="0.33" && Q_SIZE_UDIR[23]="2.50" && I_SIZE_UDIR[23]="0.50" 
NAME[24]="coPapersDBLP"      && Q_SIZE_DIR[24]="12.0" && I_SIZE_DIR[24]="0.60" && Q_SIZE_UDIR[24]="40.0" && I_SIZE_UDIR[24]="0.60" 
NAME[25]="coPapersCiteseer"  && Q_SIZE_DIR[25]="12.0" && I_SIZE_DIR[25]="0.60" && Q_SIZE_UDIR[25]="40.0" && I_SIZE_UDIR[25]="0.60" 
NAME[26]="citationCiteseer"  && Q_SIZE_DIR[26]="0.50" && I_SIZE_DIR[26]="0.20" && Q_SIZE_UDIR[26]="4.00" && I_SIZE_UDIR[26]="0.60" 
NAME[27]="preferentialAttachment" && Q_SIZE_DIR[27]="2.5" && I_SIZE_DIR[27]="0.75" && Q_SIZE_UDIR[27]="7.5" && I_SIZE_UDIR[27]="0.60" 
NAME[28]="soc-LiveJournal1"  && Q_SIZE_DIR[28]="12.0" && I_SIZE_DIR[28]="1.00" && Q_SIZE_UDIR[28]="12.0" && I_SIZE_UDIR[28]="1.00" 

NAME[29]="roadNet-CA"        && Q_SIZE_DIR[29]="0.01" && I_SIZE_DIR[29]="0.01" && Q_SIZE_UDIR[29]="0.01" && I_SIZE_UDIR[29]="0.01" 
NAME[30]="belgium_osm"       && Q_SIZE_DIR[30]="0.01" && I_SIZE_DIR[30]="0.01" && Q_SIZE_UDIR[30]="0.01" && I_SIZE_UDIR[30]="0.01" 
NAME[31]="netherlands_osm"   && Q_SIZE_DIR[31]="0.01" && I_SIZE_DIR[31]="0.01" && Q_SIZE_UDIR[31]="0.01" && I_SIZE_UDIR[31]="0.01" 
NAME[32]="italy_osm"         && Q_SIZE_DIR[32]="0.01" && I_SIZE_DIR[32]="0.01" && Q_SIZE_UDIR[32]="0.01" && I_SIZE_UDIR[32]="0.01" 
NAME[33]="luxembourg_osm"    && Q_SIZE_DIR[33]="0.01" && I_SIZE_DIR[33]="0.01" && Q_SIZE_UDIR[33]="0.01" && I_SIZE_UDIR[33]="0.01" 
NAME[34]="great-britain_osm" && Q_SIZE_DIR[34]="0.01" && I_SIZE_DIR[34]="0.01" && Q_SIZE_UDIR[34]="0.01" && I_SIZE_UDIR[34]="0.01" 
NAME[35]="germany_osm"       && Q_SIZE_DIR[35]="0.01" && I_SIZE_DIR[35]="0.01" && Q_SIZE_UDIR[35]="0.01" && I_SIZE_UDIR[35]="0.01" 
NAME[36]="asia_osm"          && Q_SIZE_DIR[36]="0.02" && I_SIZE_DIR[36]="0.02" && Q_SIZE_UDIR[36]="0.02" && I_SIZE_UDIR[36]="0.02" 
NAME[37]="europe_osm"        && Q_SIZE_DIR[37]="0.02" && I_SIZE_DIR[37]="0.02" && Q_SIZE_UDIR[37]="0.02" && I_SIZE_UDIR[37]="0.02" 
NAME[38]="road_usa"          && Q_SIZE_DIR[38]="0.01" && I_SIZE_DIR[38]="0.01" && Q_SIZE_UDIR[38]="0.01" && I_SIZE_UDIR[38]="0.01" 
NAME[39]="road_central"      && Q_SIZE_DIR[39]="0.01" && I_SIZE_DIR[39]="0.01" && Q_SIZE_UDIR[39]="0.01" && I_SIZE_UDIR[39]="0.01" 

NAME[40]="webbase-1M"        && Q_SIZE_DIR[40]="0.30" && I_SIZE_DIR[40]="0.15" && Q_SIZE_UDIR[40]="10.0" && I_SIZE_UDIR[40]="0.50" 
NAME[41]="tweets"            && Q_SIZE_DIR[41]="3.00" && I_SIZE_DIR[41]="0.75" && Q_SIZE_UDIR[41]="3.00" && I_SIZE_UDIR[41]="0.75" 
NAME[42]="bitcoin"           && Q_SIZE_DIR[42]="3.00" && I_SIZE_DIR[42]="1.00" && Q_SIZE_UDIR[42]="6.00" && I_SIZE_UDIR[42]="1.00" 
NAME[43]="caidaRouterLevel"  && Q_SIZE_DIR[43]="1.00" && I_SIZE_DIR[43]="0.30" && Q_SIZE_UDIR[43]="3.60" && I_SIZE_UDIR[43]="0.40" 

for i in {0..43} 
do
    for j in 0 1 2 3 4 6
    do
        if [ "$j" -eq "0" ] || [ "$j" -eq "1" ] || [ "$j" -eq "4" ] || [ "$j" -eq "5" ]; then
            echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_DIR[$i]} --in-sizing=${I_SIZE_DIR[$i]} "> eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARK[$j]}.txt"
            $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_DIR[$i]} --in-sizing=${I_SIZE_DIR[$i]} > eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARK[$j]}.txt
        else
            echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_UDIR[$i]} --in-sizing=${I_SIZE_UDIR[$i]} "> eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARK[$j]}.txt"
            $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_UDIR[$i]} --in-sizing=${I_SIZE_UDIR[$i]} > eval/$SUFFIX/${NAME[$i]}.$SUFFIX${MARK[$j]}.txt
        fi
        sleep 1
    done
done
