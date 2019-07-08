#!/bin/bash

# examples.sh

# --
# Small datasets

seq 39 > chesapeake.values

# uniform random
./bin/test_rw* --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --walk-mode 0 --seed 123

# greedy
./bin/test_rw* --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1

# uniform (no save path, no reference)
./bin/test_rw* --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 0 --store-walks 0 --quick --seed 123

# greedy (no save path)
./bin/test_rw* --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --store-walks 0 --walk-length 100 --quick

# --
# Should match reference

# dummy, undirected
./bin/test_rw* --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --walk-length 5

# dummy, directed
./bin/test_rw* --graph-type market --graph-file dir_chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --walk-length 100 --undirected=0

# dummy, undirected
./bin/test_rw* --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 2 --walk-length 20 --seed 123


# real graph
python ~/edgelist2mtx.py \
    --inpath /home/bjohnson/projects/hive/cpp/graphsearch/dataset/gs_twitter.edgelist \
    --outpath undir_gs_twitter

cp /home/bjohnson/projects/hive/cpp/graphsearch/dataset/gs_twitter.values gs_twitter.values 

# --
# HIVE twitter graph

# directed, greedy
# change `undir_gs_twitter.mtx` header to `general` to create `dir_gs_twitter.mtx`
# can run this for a high `--walk-length`, because the walks just go uphill and terminate
# average walk length is very short
./bin/test_rw* --graph-type market --graph-file dir_gs_twitter.mtx \
    --node-value-path gs_twitter.values \
    --walk-mode 1 \
    --walk-length 32 \
    --undirected=0 \
    --store-walks 0 \
    --quick \
    --num-runs 10

# # undirected, greedy
# # This is much more expensive, because the walks don't terminate
# # They actually go to the peak, and the bounce back and forth stupidly
# # `total_neighbors_seen` overflows
# ./bin/test_rw* --graph-type market --graph-file undir_gs_twitter.mtx \
#     --node-value-path gs_twitter.values \
#     --walk-mode 1 \
#     --walk-length 100 \
#     --store-walks 0 \
#     --quick \
#     --num-runs 10

# undirected, random
# waaay faster than the CPU reference implementation
./bin/test_rw* --graph-type market --graph-file undir_gs_twitter.mtx \
    --node-value-path gs_twitter.values \
    --walk-mode 0 \
    --walk-length 128 \
    --store-walks 0 \
    --quick \
    --num-runs 10 \
    --seed 123

# directed, random
./bin/test_rw* --graph-type market --graph-file dir_gs_twitter.mtx \
    --node-value-path gs_twitter.values \
    --walk-mode 0 \
    --walk-length 128 \
    --undirected=0 \
    --store-walks 0 \
    --quick \
    --num-runs 10 \
    --seed 123

# --
# Larger datasets

python random-values.py 1139905 > hollywood.values
DATAPATH="../../dataset/large/hollywood-2009/hollywood-2009.mtx"

./bin/test_rw* --graph-type market --graph-file $DATAPATH \
    --walk-mode 0 \
    --walk-length 3000 \
    --seed 123 \
    --store-walks 0 --quick
