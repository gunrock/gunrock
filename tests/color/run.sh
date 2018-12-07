#!/bin/bash
read -p "--JPL: " JPL
read -p "--no_conflict: " no_conflict
read -p "--seed: " seed
read -p "--user_iter: " usr_iter
./bin/test_color_10.0_x86_64 --graph-type=market \
--graph-file=../../dataset/small/test_cc.mtx \
--JPL=$JPL \
--no_conflict=$no_conflict \
--seed=$seed \
--usr_iter=$usr_iter
