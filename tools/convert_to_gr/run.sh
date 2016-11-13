#!/bin/bash

EXECUTION=./convert_to_gr_7.0_x86_64

#put OS and Device type here
DATADIR="/data/gunrock_dataset/large"
OUTDIR="/data-2/galois_dataset"

NAME[ 0]="soc-orkut"
NAME[ 1]="ljournal-2008"
NAME[ 2]="hollywood-2009"
NAME[ 3]="indochina-2004"
NAME[ 4]="kron_g500-logn21"
NAME[ 5]="kron_g500-logn20"
NAME[ 6]="delaunay_n24"
NAME[ 7]="rgg_n_2_24_s0"
NAME[ 8]="europe_osm"

for i in {0..8} 
do
    echo $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx --undirected --include-header --keep-num --edge_value --output-filename=$OUTDIR/${NAME[$i]}_ud_ev.mtx
    $EXECUTION market $DATADIR/${NAME[$i]}/${NAME[$i]}.mtx --undirected --include-header --keep-num --edge_value --output-filename=$OUTDIR/${NAME[$i]}_ud_ev.mtx
    echo /data/Compare/Galois-2.2.1/release/tools/graph-convert/graph-convert -mtx2floatgr $OUTDIR/${NAME[$i]}_ud_ev.mtx $OUTDIR/${NAME[$i]}_ud_ev.gr
    /data/Compare/Galois-2.2.1/release/tools/graph-convert/graph-convert -mtx2floatgr $OUTDIR/${NAME[$i]}_ud_ev.mtx $OUTDIR/${NAME[$i]}_ud_ev.gr
done
