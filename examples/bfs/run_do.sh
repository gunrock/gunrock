#!/bin/bash

OPTIONS=""
OPTIONS="${OPTIONS} --src=random --num-runs=10"
#OPTIONS="${OPTIONS} --queue-factor=6.5"
#OPTIONS="${OPTIONS} --in-sizing=4"
#OPTIONS="${OPTIONS} --undirected=true,false"
#OPTIONS="${OPTIONS} --idempotence=true,false"
#OPTIONS="${OPTIONS} --mark-pred=true,false"
#OPTIONS="${OPTIONS} --direction-optimized=true,false"
OPTIONS="${OPTIONS} --direction-optimized=true"
OPTIONS="${OPTIONS} --advance-mode=TWC,LB,LB_CULL"
#OPTIONS="${OPTIONS} --do-a=0.000001,0.000002,0.000005,0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000"
OPTIONS="${OPTIONS} --do-b=0.000001,0.000002,0.000005,0.00001,0.00002,0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10,20,50,100,200,500,1000,2000,5000,10000,20000"
EXECUTION="./bin/test_bfs_10.0_x86_64"
DATADIR="/data/gunrock_dataset/large"

NAME[ 0]="soc-twitter-2010" && OPT[ 0]="--queue-factor=1.1 --do-a=0.02,0.002,0.005" #DO_A[ 0]="0.005" && DO_B[ 0]="0.1"
NAME[ 1]="hollywood-2009"   && OPT[ 1]="--do-a=0.01,0.02,0.05,0.1,0.2" #DO_A[ 1]="0.006" && DO_B[ 1]="0.1"
#0.1, 0.1 for single GPU
NAME[ 2]="soc-sinaweibo"    && OPT[ 2]="--queue-factor=1.1 --undirected=true,false --do-a=0.00002,0.00005" #DO_A[ 2]="0.00005" && DO_B[ 2]="0.1"
NAME[ 3]="soc-orkut"        && OPT[ 3]="--do-a=0.012,0.2,0.5,5" #DO_A[ 3]="0.012" && DO_B[ 3]="0.1"
NAME[ 4]="soc-LiveJournal1" && OPT[ 4]="--undirected=true,false --do-a=0.05,0.02,0.200" #DO_A[ 4]="0.200" && DO_B[ 4]="0.1"
NAME[ 5]="uk-2002"          && OPT[ 5]="--undirected=true,false --do-a=5,20,200,5000,20000" #DO_A[ 5]="16"    && DO_B[ 5]="100"
NAME[ 6]="uk-2005"          && OPT[ 6]="--undirected=true,false --do-a=5,20,200,5000,20000" #DO_A[ 6]="10"    && DO_B[ 6]="20"
NAME[ 7]="webbase-2001"     && OPT[ 7]="--queue-factor=2 --undirected=true,false --do-a=10000,100,50,5,2,0.02" ##DO_A[ 7]="100"   && DO_B[ 7]="100"
#2, 1.2 for memory
NAME[ 8]="indochina-2004"   && OPT[ 8]="--undirected=true,false --do-a=500,2000,200,10000,1000" #DO_A[ 8]="100"   && DO_B[ 8]="100"
NAME[ 9]="arabic-2005"      && OPT[ 9]="--queue-factor=1.1 --undirected=true,false --do-a=500,2000,200,10000,1000,100" ##DO_A[ 9]="100"   && DO_B[ 9]="100"
NAME[10]="europe_osm"       && OPT[10]="--do-a=2.5,1,20000" ##DO_A[10]="2.5"   && DO_B[10]="15"
NAME[11]="asia_osm"         && OPT[11]="--queue-factor=0.01 --do-a=20000,-1,1.5" #DO_A[11]="1.5"   && DO_B[11]="10"
NAME[12]="germany_osm"      && OPT[12]="--do-a=0.5,20,1000,0.05" #DO_A[12]="1.5"   && DO_B[12]="10"
NAME[13]="road_usa"          && OPT[13]="--queue-factor=0.01 --do-a=20000,500,5000,10,0.5" #DO_A[13]="1.0"   && DO_B[13]="10"
NAME[14]="road_central"      && OPT[14]="--queue-factor=0.01 --do-a=0.00001,20000,0.1"  #DO_A[14]="1.2" && DO_B[14]="10"
NAME[15]="delaunay_n24"      && OPT[15]="--do-a=0.1,1,0.05"
NAME[16]="kron_g500-logn21"  && OPT[16]="--do-a=0.00005,0.00001,0.000002,0.000001,0.000005,0.00002"
NAME[17]="coAuthorsDBLP"     && OPT[17]="--do-a=0.005,0.02"
NAME[18]="coAuthorsCiteseer" && OPT[18]="--do-a=0.02,0.05,0.2"
NAME[19]="coPapersDBLP"      && OPT[19]="--do-a=0.1,1,0.05"
NAME[20]="coPapersCiteseer"  && OPT[20]="--do-a=0.2,0.5,0.05"
NAME[21]="citationCiteseer"  && OPT[21]="--do-a=0.2,0.5,1" 
NAME[22]="preferentialAttachment" && OPT[22]="--do-a=0.005,0.001,0.0005,0.002"
NAME[23]="webbase-1M"        && OPT[23]="--undirected=true,false --do-a=10000,100,50,5,2,0.02"
#NAME[24]="tweets"
#NAME[25]="bitcoin"
NAME[26]="caidaRouterLevel" && OPT[26]="--do-a=0.05,0.1,0.2,0.5" 

GRAPH[27]="rmat --graph-scale=22 --graph-edgefactor=64" && NAME[27]="rmat_n22_e64" && OPT[27]="--undirected=true,false --do-a=0.000005,0.000002,0.00001,0.05"
GRAPH[28]="rmat --graph-scale=23 --graph-edgefactor=32" && NAME[28]="rmat_n23_e32" && OPT[28]="--undirected=true,false --do-a=0.000005,0.000002,0.00001,0.05"
GRAPH[29]="rmat --graph-scale=24 --graph-edgefactor=16" && NAME[29]="rmat_n24_e16" && OPT[29]="--undirected=true,false --do-a=0.000005,0.000002,0.00001,0.05" 
GRAPH[30]="rgg --graph-scale=24 --rgg-thfactor=0.00548" && NAME[30]="rgg_n24_0.000548" && OPT[30]="--undirected=true,false --queue-factor=0.25 --do-a=0.1,1,0.05"

for d in 1 #{1..4}
do
    SUFFIX="GUNROCK_v1-0-0_TitanVx${d}"
    mkdir -p eval/$SUFFIX
    DEVICE="0"
    for i in {1..8}
    do
        if [ "$i" -lt "$d" ]; then
            DEVICE=$DEVICE",$i"
        fi
    done

    for i in {10..30}
    do
        if [ "$i" -lt "27" ]; then
            GRAPH_="market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx"
        else
            GRAPH_="${GRAPH[$i]}"
        fi

        echo $EXECUTION ${GRAPH_} $OPTIONS ${OPT[$i]} --device=$DEVICE --jsondir=./eval/$SUFFIX "> ./eval/$SUFFIX/${NAME[$i]}.do2.txt"
             $EXECUTION ${GRAPH_} $OPTIONS ${OPT[$i]} --device=$DEVICE --jsondir=./eval/$SUFFIX  > ./eval/$SUFFIX/${NAME[$i]}.do2.txt
        sleep 1
    done
done

