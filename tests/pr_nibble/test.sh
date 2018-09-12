#!/bin/bash

# test.sh

./bin/test_pr_nibble_9.1_x86_64 \
    --graph-type market \
    --graph-file ../../dataset/small/chesapeake.mtx \
    --src 0 \
    --max-iter 1


./bin/test_pr_nibble_9.1_x86_64 \
    --graph-type market \
    --graph-file jhu \
    --src 0 \
    --max-iter 20

