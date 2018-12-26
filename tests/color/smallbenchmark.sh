#!/bin/bash
echo "[0] test on small graph"
echo "[1] test on offshore"
read -p "Option: " option
if (($option == "0"))
then
    address="../../dataset/small/test_cc.mtx"
fi

if (($option == "1"))
then
    address="/data-2/topc-datasets/gc-data/offshore/offshore.mtx"
fi

./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=$address \
    --JPL=true \
    --no-conflict=0 \
    --seed=123 \
    --user-iter=0 \
    --hash-size=0 \
    --quick=false \
    --test-run=true \
    --undirected

