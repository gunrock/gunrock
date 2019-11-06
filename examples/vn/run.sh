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

OPTION[0]="--src=largestdegree --device=2,3 --partition-method=biasrandom --grid-size=768"
#OPTION[0]="" #directed and do not mark-pred"
OPTION[1]=${OPTION[0]}" --undirected" #undirected and do not mark-pred"
OPTION[2]=${OPTION[0]}" --mark-path"  #directed and mark-pred"
OPTION[3]=${OPTION[1]}" --mark-path" #undirected and mark-pred"
#OPTION[4]=${OPTION[0]}" --idempotence"
#OPTION[5]=${OPTION[1]}" --idempotence"
#OPTION[6]=${OPTION[2]}" --idempotence"
#OPTION[7]=${OPTION[3]}" --idempotence"

MARK[0]=""
MARK[1]=${MARK[0]}"_undir"
MARK[2]=${MARK[0]}"_markpath"
MARK[3]=${MARK[1]}"_markpath"
#MARK[4]=${MARK[0]}".idempotence"
#MARK[5]=${MARK[1]}".idempotence"
#MARK[6]=${MARK[2]}".idempotence"
#MARK[7]=${MARK[3]}".idempotence"

#put OS and Device type here
EXCUTION=$exe_file
DATADIR="/data/gunrock_dataset/large"

NAME[ 0]="ak2010"            && Q_SIZE_DIR[ 0]="0.05" && I_SIZE_DIR[ 0]="0.02" && Q_SIZE_UDIR[ 0]="3.00" && I_SIZE_UDIR[ 0]="1.00"

NAME[ 1]="delaunay_n10"      && Q_SIZE_DIR[ 1]="2.00" && I_SIZE_DIR[ 1]="0.40" && Q_SIZE_UDIR[ 1]="2.50" && I_SIZE_UDIR[ 1]="0.60"
NAME[ 2]="delaunay_n11"      && Q_SIZE_DIR[ 2]="2.00" && I_SIZE_DIR[ 2]="0.40" && Q_SIZE_UDIR[ 2]="2.50" && I_SIZE_UDIR[ 2]="0.50"
NAME[ 3]="delaunay_n12"      && Q_SIZE_DIR[ 3]="2.00" && I_SIZE_DIR[ 3]="0.40" && Q_SIZE_UDIR[ 3]="2.00" && I_SIZE_UDIR[ 3]="0.50"
NAME[ 4]="delaunay_n13"      && Q_SIZE_DIR[ 4]="0.25" && I_SIZE_DIR[ 4]="0.15" && Q_SIZE_UDIR[ 4]="1.40" && I_SIZE_UDIR[ 4]="0.30"
NAME[ 5]="delaunay_n14"      && Q_SIZE_DIR[ 5]="0.25" && I_SIZE_DIR[ 5]="0.15" && Q_SIZE_UDIR[ 5]="1.20" && I_SIZE_UDIR[ 5]="0.30"
NAME[ 6]="delaunay_n15"      && Q_SIZE_DIR[ 6]="0.25" && I_SIZE_DIR[ 6]="0.15" && Q_SIZE_UDIR[ 6]="1.20" && I_SIZE_UDIR[ 6]="0.30"
NAME[ 7]="delaunay_n16"      && Q_SIZE_DIR[ 7]="0.25" && I_SIZE_DIR[ 7]="0.15" && Q_SIZE_UDIR[ 7]="0.90" && I_SIZE_UDIR[ 7]="0.20"
NAME[ 8]="delaunay_n17"      && Q_SIZE_DIR[ 8]="0.25" && I_SIZE_DIR[ 8]="0.15" && Q_SIZE_UDIR[ 8]="0.80" && I_SIZE_UDIR[ 8]="0.20"
NAME[ 9]="delaunay_n18"      && Q_SIZE_DIR[ 9]="0.25" && I_SIZE_DIR[ 9]="0.15" && Q_SIZE_UDIR[ 9]="0.80" && I_SIZE_UDIR[ 9]="0.20"
NAME[10]="delaunay_n19"      && Q_SIZE_DIR[10]="0.25" && I_SIZE_DIR[10]="0.15" && Q_SIZE_UDIR[10]="0.80" && I_SIZE_UDIR[10]="0.20"
NAME[11]="delaunay_n20"      && Q_SIZE_DIR[11]="0.25" && I_SIZE_DIR[11]="0.15" && Q_SIZE_UDIR[11]="0.80" && I_SIZE_UDIR[11]="0.20"
NAME[12]="delaunay_n21"      && Q_SIZE_DIR[12]="0.25" && I_SIZE_DIR[12]="0.15" && Q_SIZE_UDIR[12]="0.80" && I_SIZE_UDIR[12]="0.20"
NAME[13]="delaunay_n22"      && Q_SIZE_DIR[13]="0.25" && I_SIZE_DIR[13]="0.15" && Q_SIZE_UDIR[13]="0.80" && I_SIZE_UDIR[13]="0.20"
NAME[14]="delaunay_n23"      && Q_SIZE_DIR[14]="0.25" && I_SIZE_DIR[14]="0.15" && Q_SIZE_UDIR[14]="0.80" && I_SIZE_UDIR[14]="0.20"
NAME[15]="delaunay_n24"      && Q_SIZE_DIR[15]="0.25" && I_SIZE_DIR[15]="0.15" && Q_SIZE_UDIR[15]="0.80" && I_SIZE_UDIR[15]="0.20"

NAME[16]="kron_g500-logn16"  && Q_SIZE_DIR[16]="180 " && I_SIZE_DIR[16]="1.50" && Q_SIZE_UDIR[16]="380 " && I_SIZE_UDIR[16]="1.75" 
NAME[17]="kron_g500-logn17"  && Q_SIZE_DIR[17]="180 " && I_SIZE_DIR[17]="1.50" && Q_SIZE_UDIR[17]="380 " && I_SIZE_UDIR[17]="1.75" 
NAME[18]="kron_g500-logn18"  && Q_SIZE_DIR[18]="180 " && I_SIZE_DIR[18]="1.50" && Q_SIZE_UDIR[18]="380 " && I_SIZE_UDIR[18]="1.75" 
NAME[19]="kron_g500-logn19"  && Q_SIZE_DIR[19]="180 " && I_SIZE_DIR[19]="1.50" && Q_SIZE_UDIR[19]="380 " && I_SIZE_UDIR[19]="1.75" 
NAME[20]="kron_g500-logn20"  && Q_SIZE_DIR[20]="180 " && I_SIZE_DIR[20]="1.50" && Q_SIZE_UDIR[20]="380 " && I_SIZE_UDIR[20]="1.75" 
NAME[21]="kron_g500-logn21"  && Q_SIZE_DIR[21]="180 " && I_SIZE_DIR[21]="1.50" && Q_SIZE_UDIR[21]="380 " && I_SIZE_UDIR[21]="1.75" 

NAME[22]="coAuthorsDBLP"     && Q_SIZE_DIR[22]="1.40" && I_SIZE_DIR[22]="0.50" && Q_SIZE_UDIR[22]="10.0" && I_SIZE_UDIR[22]="1.20" 
NAME[23]="coAuthorsCiteseer" && Q_SIZE_DIR[23]="1.40" && I_SIZE_DIR[23]="0.50" && Q_SIZE_UDIR[23]="13.0" && I_SIZE_UDIR[23]="1.20" 
NAME[24]="coPapersDBLP"      && Q_SIZE_DIR[24]="42.0" && I_SIZE_DIR[24]="1.50" && Q_SIZE_UDIR[24]="210 " && I_SIZE_UDIR[24]="2.40" 
NAME[25]="coPapersCiteseer"  && Q_SIZE_DIR[25]="42.0" && I_SIZE_DIR[25]="1.50" && Q_SIZE_UDIR[25]="210 " && I_SIZE_UDIR[25]="2.40" 
NAME[26]="citationCiteseer"  && Q_SIZE_DIR[26]="1.40" && I_SIZE_DIR[26]="0.50" && Q_SIZE_UDIR[26]="16.0" && I_SIZE_UDIR[26]="1.40" 
NAME[27]="preferentialAttachment" && Q_SIZE_DIR[27]="6.0" && I_SIZE_DIR[27]="1.20" && Q_SIZE_UDIR[27]="20.0" && I_SIZE_UDIR[27]="1.50" 
NAME[28]="soc-LiveJournal1"  && Q_SIZE_DIR[28]="30.0" && I_SIZE_DIR[28]="2.00" && Q_SIZE_UDIR[28]="50.0" && I_SIZE_UDIR[28]="2.00" 

NAME[29]="roadNet-CA"        && Q_SIZE_DIR[29]="0.01" && I_SIZE_DIR[29]="0.01" && Q_SIZE_UDIR[29]="0.25" && I_SIZE_UDIR[29]="0.01" 
NAME[30]="belgium_osm"       && Q_SIZE_DIR[30]="0.04" && I_SIZE_DIR[30]="0.04" && Q_SIZE_UDIR[30]="0.05" && I_SIZE_UDIR[30]="0.04" 
NAME[31]="netherlands_osm"   && Q_SIZE_DIR[31]="0.01" && I_SIZE_DIR[31]="0.01" && Q_SIZE_UDIR[31]="0.02" && I_SIZE_UDIR[31]="0.01" 
NAME[32]="italy_osm"         && Q_SIZE_DIR[32]="0.01" && I_SIZE_DIR[32]="0.01" && Q_SIZE_UDIR[32]="0.02" && I_SIZE_UDIR[32]="0.01" 
NAME[33]="luxembourg_osm"    && Q_SIZE_DIR[33]="0.04" && I_SIZE_DIR[33]="0.04" && Q_SIZE_UDIR[33]="0.04" && I_SIZE_UDIR[33]="0.04" 
NAME[34]="great-britain_osm" && Q_SIZE_DIR[34]="0.01" && I_SIZE_DIR[34]="0.01" && Q_SIZE_UDIR[34]="0.02" && I_SIZE_UDIR[34]="0.01" 
NAME[35]="germany_osm"       && Q_SIZE_DIR[35]="0.01" && I_SIZE_DIR[35]="0.01" && Q_SIZE_UDIR[35]="0.02" && I_SIZE_UDIR[35]="0.01" 
NAME[36]="asia_osm"          && Q_SIZE_DIR[36]="0.02" && I_SIZE_DIR[36]="0.02" && Q_SIZE_UDIR[36]="0.03" && I_SIZE_UDIR[36]="0.02" 
NAME[37]="europe_osm"        && Q_SIZE_DIR[37]="0.02" && I_SIZE_DIR[37]="0.02" && Q_SIZE_UDIR[37]="0.02" && I_SIZE_UDIR[37]="0.02" 
NAME[38]="road_usa"          && Q_SIZE_DIR[38]="0.01" && I_SIZE_DIR[38]="0.01" && Q_SIZE_UDIR[38]="0.15" && I_SIZE_UDIR[38]="0.01" 
NAME[39]="road_central"      && Q_SIZE_DIR[39]="0.01" && I_SIZE_DIR[39]="0.01" && Q_SIZE_UDIR[39]="0.15" && I_SIZE_UDIR[39]="0.01" 

NAME[40]="webbase-1M"        && Q_SIZE_DIR[40]="0.30" && I_SIZE_DIR[40]="0.15" && Q_SIZE_UDIR[40]="18.0" && I_SIZE_UDIR[40]="1.30" 
NAME[41]="tweets"            && Q_SIZE_DIR[41]="10.0" && I_SIZE_DIR[41]="2.00" && Q_SIZE_UDIR[41]="5.50" && I_SIZE_UDIR[41]="2.00" 
NAME[42]="bitcoin"           && Q_SIZE_DIR[42]="5.00" && I_SIZE_DIR[42]="2.00" && Q_SIZE_UDIR[42]="10.0" && I_SIZE_UDIR[42]="2.00" 
NAME[43]="caidaRouterLevel"  && Q_SIZE_DIR[43]="1.50" && I_SIZE_DIR[43]="0.50" && Q_SIZE_UDIR[43]="12.0" && I_SIZE_UDIR[43]="1.20" 

F[0]="0.0" && F[1]="0.1" && F[2]="0.2" && F[3]="0.3" && F[4]="0.4" && F[5]="0.5" && F[6]="0.6" && F[7]="0.7" && F[8]="0.8" && F[9]="0.9"
F[10]="1.0"
 
for k in {0..10}
do
    #put OS and Device type here
    SUFFIX="ubuntu12.04_k40cx2_brp${F[$k]}"
    mkdir -p eval/$SUFFIX

    for i in  {0..43} 
    do
        for j in {0..3}
        do
            if [ "$j" -eq "0" ] || [ "$j" -eq "2" ] ; then
                #echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_DIR[$i]} --in-sizing=${I_SIZE_DIR[$i]} "> eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt"
                #$EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_DIR[$i]} --in-sizing=${I_SIZE_DIR[$i]} > eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt
                echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} "> eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt"
                $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} > eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt
            else
                #echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_UDIR[$i]} --in-sizing=${I_SIZE_UDIR[$i]} "> eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt"
                #$EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=${Q_SIZE_UDIR[$i]} --in-sizing=${I_SIZE_UDIR[$i]} > eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt
                echo $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} "> eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt"
                $EXCUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx ${OPTION[$j]} --queue-sizing=0.01 --in-sizing=0.01 --partition-factor=${F[$k]} > eval/$SUFFIX/${NAME[$i]}_$SUFFIX${MARK[$j]}.txt
            fi
            sleep 1
        done
    done
done

