#!/bin/bash
read -p "--JPL: " JPL
read -p "--no-conflict: " no_conflict
read -p "--seed: " seed
read -p "--user-iter: " usr_iter
read -p "--hash-size: " hash_size
read -p "--graph-file:" graph_file
read -p "--quick:" quick
read -p "--test-run: " test_run
./bin/test_color_10.0_x86_64 --graph-type=market \
--graph-file=../../dataset/small/test_cc.mtx \
--JPL=$JPL \
--no-conflict=$no_conflict \
--seed=$seed \
--usr-iter=$usr_iter \
--hash-size=$hash_size \
--quick=$quick \
--test-run=$test_run
