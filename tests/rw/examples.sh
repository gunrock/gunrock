#!/bin/bash

# examples.sh

# --
# Small datasets

python random-values.py 39 > chesapeake.values

# uniform random
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --walk-mode 0

# greedy
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1

# uniform (no save path)
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 0 --store-walks 0

# greedy (no save path)
./bin/test_rw_9.1_x86_64 --graph-type market --graph-file ../../dataset/small/chesapeake.mtx \
    --node-value-path chesapeake.values --walk-mode 1 --store-walks 0


# --
# Larger datasets

python random-values.py 1139905 > hollywood.values
DATAPATH="../../dataset/large/hollywood-2009/hollywood-2009.mtx"

./bin/test_rw_9.1_x86_64 --graph-type market --graph-file $DATAPATH \
    --walk-mode 0 \
    --walk-length 3000 \
    --quick --quiet