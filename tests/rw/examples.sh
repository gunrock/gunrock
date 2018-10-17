#!/bin/bash

# examples.sh

# --
# Small datasets

seq 39 > chesapeake.values

# uniform random
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --walk-mode 0 --seed 123

# greedy
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --seed 123

# uniform (no save path, no reference)
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 0 --store-walks 0 --quick --seed 123

# greedy (no save path)
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --store-walks 0 --walk-length 100 --quick

# --
# Should match reference

# dummy, undirected
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --walk-length 5

# dummy, directed
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file dir_chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --walk-length 5 --undirected=0

# real graph
python ~/edgelist2mtx.py \
    --inpath /home/bjohnson/projects/hive/cpp/graphsearch/dataset/gs_twitter.edgelist \
    --outpath gs_twitter

cp /home/bjohnson/projects/hive/cpp/graphsearch/dataset/gs_twitter.values gs_twitter.values 

# change gs_twitter.mtx to `general`
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file gs_twitter.mtx \
    --node-value-path gs_twitter.values \
    --walk-mode 1 \
    --walk-length 10000 \
    --undirected=0 \
    --store-walks 0 \
    --quick \
    --num-runs 10

# --
# Larger datasets

python random-values.py 1139905 > hollywood.values
DATAPATH="../../dataset/large/hollywood-2009/hollywood-2009.mtx"

./bin/test_rw_9.1_x86_64 --graph-type market --graph-file $DATAPATH \
    --walk-mode 0 \
    --walk-length 3000 \
    --seed 123 \
    --store-walks 0 --quick

# --
# HIVE dataset

cat ~/projects/hive/cpp/graphsearch/dataset/gs_twitter.scores |\
    cut -d$'\t' -f2 > twitter.scores

python ~/edgelist2mtx.py \
    --inpath ~/projects/hive/cpp/graphsearch/dataset/gs_twitter.edgelist \
    --outpath twitter

./bin/test_rw_9.1_x86_64 --graph-type market --graph-file twitter.mtx \
    --walk-mode 0 \
    --walk-length 10000 \
    --seed 123 \
    --store-walks 0 --quick