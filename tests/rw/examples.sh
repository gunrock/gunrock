#!/bin/bash

# examples.sh

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

