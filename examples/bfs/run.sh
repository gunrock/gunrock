#!/bin/bash

#get all execution files in ./bin
files=./bin/*
#split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
EXECUTION=""
#iterate over all file names to get the largest version number
for x in $arr;
do
    output=$(grep -o "[0-9]\.[0-9]" <<< "$x")
    if [ "$output" \> "$max_ver_num" ]; then
        EXECUTION=$x
    fi
done

#put OS and Device type here
SUFFIX="GUNROCK_v1-0-0_TitanV"
DATADIR="/data/gunrock_dataset/large"

ORG_OPTIONS=""
ORG_OPTIONS="$ORG_OPTIONS --num-runs=10"
ORG_OPTIONS="$ORG_OPTIONS --validation=each"
ORG_OPTIONS="$ORG_OPTIONS --device=0"
ORG_OPTIONS="$ORG_OPTIONS --jsondir=./eval/$SUFFIX"
ORG_OPTIONS="$ORG_OPTIONS --src=random"
#ORG_OPTIONS="$ORG_OPTIONS --64bit-SizeT=true,false"
#ORG_OPTIONS="$ORG_OPTIONS --64bit-VertexT=true,false"
ORG_OPTIONS="$ORG_OPTIONS --idempotence=true,false"
ORG_OPTIONS="$ORG_OPTIONS --mark-pred=true,false"
ORG_OPTIONS="$ORG_OPTIONS --direction-optimized=true,false"
ORG_OPTIONS="$ORG_OPTIONS --advance-mode=TWC,LB,LB_CULL"

NAME[ 0]="ak2010"            && OPT_DIR[ 0]="--queue-factor=0.05" && I_SIZE_DIR[ 0]="0.02" && OPT_UDIR[ 0]="--queue-factor=0.25" && I_SIZE_UDIR[ 0]="0.08"

NAME[ 1]="delaunay_n10"      && OPT_DIR[ 1]="--queue-factor=2.00" && I_SIZE_DIR[ 1]="0.40" && OPT_UDIR[ 1]="--queue-factor=2.00" && I_SIZE_UDIR[ 1]="0.40"
NAME[ 2]="delaunay_n11"      && OPT_DIR[ 2]="--queue-factor=2.00" && I_SIZE_DIR[ 2]="0.40" && OPT_UDIR[ 2]="--queue-factor=2.00" && I_SIZE_UDIR[ 2]="0.40"
NAME[ 3]="delaunay_n12"      && OPT_DIR[ 3]="--queue-factor=2.00" && I_SIZE_DIR[ 3]="0.40" && OPT_UDIR[ 3]="--queue-factor=2.00" && I_SIZE_UDIR[ 3]="0.40"
NAME[ 4]="delaunay_n13"      && OPT_DIR[ 4]="--queue-factor=0.25" && I_SIZE_DIR[ 4]="0.15" && OPT_UDIR[ 4]="--queue-factor=0.25" && I_SIZE_UDIR[ 4]="0.15"
NAME[ 5]="delaunay_n14"      && OPT_DIR[ 5]="--queue-factor=0.25" && I_SIZE_DIR[ 5]="0.15" && OPT_UDIR[ 5]="--queue-factor=0.25" && I_SIZE_UDIR[ 5]="0.15"
NAME[ 6]="delaunay_n15"      && OPT_DIR[ 6]="--queue-factor=0.25" && I_SIZE_DIR[ 6]="0.15" && OPT_UDIR[ 6]="--queue-factor=0.25" && I_SIZE_UDIR[ 6]="0.15"
NAME[ 7]="delaunay_n16"      && OPT_DIR[ 7]="--queue-factor=0.25" && I_SIZE_DIR[ 7]="0.15" && OPT_UDIR[ 7]="--queue-factor=0.25" && I_SIZE_UDIR[ 7]="0.15"
NAME[ 8]="delaunay_n17"      && OPT_DIR[ 8]="--queue-factor=0.25" && I_SIZE_DIR[ 8]="0.15" && OPT_UDIR[ 8]="--queue-factor=0.25" && I_SIZE_UDIR[ 8]="0.15"
NAME[ 9]="delaunay_n18"      && OPT_DIR[ 9]="--queue-factor=0.25" && I_SIZE_DIR[ 9]="0.15" && OPT_UDIR[ 9]="--queue-factor=0.25" && I_SIZE_UDIR[ 9]="0.15"
NAME[10]="delaunay_n19"      && OPT_DIR[10]="--queue-factor=0.25" && I_SIZE_DIR[10]="0.15" && OPT_UDIR[10]="--queue-factor=0.25" && I_SIZE_UDIR[10]="0.15"
NAME[11]="delaunay_n20"      && OPT_DIR[11]="--queue-factor=0.25" && I_SIZE_DIR[11]="0.15" && OPT_UDIR[11]="--queue-factor=0.25" && I_SIZE_UDIR[11]="0.15"
NAME[12]="delaunay_n21"      && OPT_DIR[12]="--queue-factor=0.25" && I_SIZE_DIR[12]="0.15" && OPT_UDIR[12]="--queue-factor=0.25" && I_SIZE_UDIR[12]="0.15"
NAME[13]="delaunay_n22"      && OPT_DIR[13]="--queue-factor=0.25" && I_SIZE_DIR[13]="0.15" && OPT_UDIR[13]="--queue-factor=0.25" && I_SIZE_UDIR[13]="0.15"
NAME[14]="delaunay_n23"      && OPT_DIR[14]="--queue-factor=0.25" && I_SIZE_DIR[14]="0.15" && OPT_UDIR[14]="--queue-factor=0.25" && I_SIZE_UDIR[14]="0.15"
NAME[15]="delaunay_n24"      && OPT_DIR[15]="--queue-factor=0.25" && I_SIZE_DIR[15]="0.15" && OPT_UDIR[15]="--queue-factor=0.25" && I_SIZE_UDIR[15]="0.15"

NAME[16]="kron_g500-logn16"  && OPT_DIR[16]="--queue-factor=75.0" && I_SIZE_DIR[16]="1.50" && OPT_UDIR[16]="--queue-factor=150 " && I_SIZE_UDIR[16]="1.50" 
NAME[17]="kron_g500-logn17"  && OPT_DIR[17]="--queue-factor=75.0" && I_SIZE_DIR[17]="1.50" && OPT_UDIR[17]="--queue-factor=150 " && I_SIZE_UDIR[17]="1.50" 
NAME[18]="kron_g500-logn18"  && OPT_DIR[18]="--queue-factor=75.0" && I_SIZE_DIR[18]="1.50" && OPT_UDIR[18]="--queue-factor=150 " && I_SIZE_UDIR[18]="1.50" 
NAME[19]="kron_g500-logn19"  && OPT_DIR[19]="--queue-factor=75.0" && I_SIZE_DIR[19]="1.50" && OPT_UDIR[19]="--queue-factor=150 " && I_SIZE_UDIR[19]="1.50" 
NAME[20]="kron_g500-logn20"  && OPT_DIR[20]="--queue-factor=75.0" && I_SIZE_DIR[20]="1.50" && OPT_UDIR[20]="--queue-factor=150 " && I_SIZE_UDIR[20]="1.50" 
NAME[21]="kron_g500-logn21"  && OPT_DIR[21]="--queue-factor=75.0" && I_SIZE_DIR[21]="1.50" && OPT_UDIR[21]="--queue-factor=150 " && I_SIZE_UDIR[21]="1.50" 

NAME[22]="coAuthorsDBLP"     && OPT_DIR[22]="--queue-factor=0.60" && I_SIZE_DIR[22]="0.33" && OPT_UDIR[22]="--queue-factor=2.50" && I_SIZE_UDIR[22]="0.50" 
NAME[23]="coAuthorsCiteseer" && OPT_DIR[23]="--queue-factor=0.60" && I_SIZE_DIR[23]="0.33" && OPT_UDIR[23]="--queue-factor=2.50" && I_SIZE_UDIR[23]="0.50" 
NAME[24]="coPapersDBLP"      && OPT_DIR[24]="--queue-factor=20.0" && I_SIZE_DIR[24]="1.50" && OPT_UDIR[24]="--queue-factor=60.0" && I_SIZE_UDIR[24]="1.50" 
NAME[25]="coPapersCiteseer"  && OPT_DIR[25]="--queue-factor=20.0" && I_SIZE_DIR[25]="1.50" && OPT_UDIR[25]="--queue-factor=60.0" && I_SIZE_UDIR[25]="1.50" 
NAME[26]="citationCiteseer"  && OPT_DIR[26]="--queue-factor=0.50" && I_SIZE_DIR[26]="0.20" && OPT_UDIR[26]="--queue-factor=4.00" && I_SIZE_UDIR[26]="0.60" 
NAME[27]="preferentialAttachment" && OPT_DIR[27]="--queue-factor=2.5" && I_SIZE_DIR[27]="0.75" && OPT_UDIR[27]="--queue-factor=7.5" && I_SIZE_UDIR[27]="0.60" 
NAME[28]="soc-LiveJournal1"  && OPT_DIR [28]="--queue-factor=20.0 --do-a=0.200 --do-b=0.1" && I_SIZE_DIR[28]="2.00" \
                             && OPT_UDIR[28]="--queue-factor=20.0" && I_SIZE_UDIR[28]="2.00" 
NAME[44]="soc-twitter-2010"  && OPT_DIR [44]="--do-a=0.005 --do-b=0.1"
                             && OPT_UDIR[44]=""
NAME[45]="hollywood-2009"    && OPT_DIR [44]="

NAME[29]="roadNet-CA"        && OPT_DIR[29]="--queue-factor=0.01" && I_SIZE_DIR[29]="0.01" && OPT_UDIR[29]="--queue-factor=0.01" && I_SIZE_UDIR[29]="0.01" 
NAME[30]="belgium_osm"       && OPT_DIR[30]="--queue-factor=0.01" && I_SIZE_DIR[30]="0.01" && OPT_UDIR[30]="--queue-factor=0.01" && I_SIZE_UDIR[30]="0.01" 
NAME[31]="netherlands_osm"   && OPT_DIR[31]="--queue-factor=0.01" && I_SIZE_DIR[31]="0.01" && OPT_UDIR[31]="--queue-factor=0.01" && I_SIZE_UDIR[31]="0.01" 
NAME[32]="italy_osm"         && OPT_DIR[32]="--queue-factor=0.01" && I_SIZE_DIR[32]="0.01" && OPT_UDIR[32]="--queue-factor=0.01" && I_SIZE_UDIR[32]="0.01" 
NAME[33]="luxembourg_osm"    && OPT_DIR[33]="--queue-factor=0.01" && I_SIZE_DIR[33]="0.01" && OPT_UDIR[33]="--queue-factor=0.01" && I_SIZE_UDIR[33]="0.01" 
NAME[34]="great-britain_osm" && OPT_DIR[34]="--queue-factor=0.01" && I_SIZE_DIR[34]="0.01" && OPT_UDIR[34]="--queue-factor=0.01" && I_SIZE_UDIR[34]="0.01" 
NAME[35]="germany_osm"       && OPT_DIR[35]="--queue-factor=0.01" && I_SIZE_DIR[35]="0.01" && OPT_UDIR[35]="--queue-factor=0.01" && I_SIZE_UDIR[35]="0.01" 
NAME[36]="asia_osm"          && OPT_DIR[36]="--queue-factor=0.02" && I_SIZE_DIR[36]="0.02" && OPT_UDIR[36]="--queue-factor=0.02" && I_SIZE_UDIR[36]="0.02" 
NAME[37]="europe_osm"        && OPT_DIR[37]="--queue-factor=0.02" && I_SIZE_DIR[37]="0.02" && OPT_UDIR[37]="--queue-factor=0.02" && I_SIZE_UDIR[37]="0.02" 
NAME[38]="road_usa"          && OPT_DIR[38]="--queue-factor=0.01" && I_SIZE_DIR[38]="0.01" && OPT_UDIR[38]="--queue-factor=0.01" && I_SIZE_UDIR[38]="0.01" 
NAME[39]="road_central"      && OPT_DIR[39]="--queue-factor=0.01" && I_SIZE_DIR[39]="0.01" && OPT_UDIR[39]="--queue-factor=0.01" && I_SIZE_UDIR[39]="0.01" 

NAME[40]="webbase-1M"        && OPT_DIR[40]="--queue-factor=0.30" && I_SIZE_DIR[40]="0.15" && OPT_UDIR[40]="--queue-factor=10.0" && I_SIZE_UDIR[40]="0.50" 
NAME[41]="tweets"            && OPT_DIR[41]="--queue-factor=5.00" && I_SIZE_DIR[41]="2.00" && OPT_UDIR[41]="--queue-factor=5.00" && I_SIZE_UDIR[41]="2.00" 
NAME[42]="bitcoin"           && OPT_DIR[42]="--queue-factor=5.00" && I_SIZE_DIR[42]="2.00" && OPT_UDIR[42]="--queue-factor=10.0" && I_SIZE_UDIR[42]="2.00" 
NAME[43]="caidaRouterLevel"  && OPT_DIR[43]="--queue-factor=1.00" && I_SIZE_DIR[43]="0.30" && OPT_UDIR[43]="--queue-factor=3.60" && I_SIZE_UDIR[43]="0.40" 

mkdir -p eval/$SUFFIX

for i in {0..43}; do
    for undirected in "true" "false"; do
        OPTIONS=$ORG_OPTIONS
        MARKS=""
        if [ "$undirected" = "true" ]; then
            OPTIONS="$OPTIONS --queue-factor=${Q_SIZE_UDIR[$i]}"
            MARKS="UDIR"
        else
            OPTIONS="$OPTIONS --queue-factor=${Q_SIZE_DIR[$i]}"
            MARKS="DIR"
        fi

        echo $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTIONS "> ./eval/$SUFFIX/${NAME[$i]}.$SUFFIX.${MARKS}.txt"
             $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx $OPTIONS > ./eval/$SUFFIX/${NAME[$i]}.$SUFFIX.${MARKS}.txt
        sleep 1
    done
done
