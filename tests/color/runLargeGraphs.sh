#!/bin/bash
read -p "--JPL: " JPL
read -p "--no-conflict: " no_conflict
read -p "--seed: " seed
read -p "--user-iter: " usr_iter
read -p "--hash-size: " hash_size
read -p "--graph-file:" graph_file
read -p "--quick:" quick
read -p "--test-run: " test_run

#address
addr1="offshore/offshore.mtx"
addr2="af_shell3/af_shell3.mtx"
addr3="parabolic_fem/parabolic_fem.mtx"
addr4="apache2/apache2.mtx"
addr5="ecology2/ecology2.mtx"
addr6="thermal2/thermal2.mtx"
addr7="G3_circuit/G3_circuit.mtx"
addr8="FEM_3D_thermal2/FEM_3D_thermal2.mtx"
addr9="thermomech_dK/thermomech_dK.mtx"
addr10="ASIC_320ks/ASIC_320ks.mtx"
addr11="cage13/cage13.mtx"
addr12="atmosmodd/atmosmodd.mtx"

for i in `seq 1 12`;
do
  ./bin/test_color_10.0_x86_64 --graph-type=market \
  --graph-file="/data-2/topc-datasets/gc-data/ASIC_320ks/${addr${i}}"  \
  --JPL=$JPL \
  --no_conflict=$no_conflict \
  --seed=$seed \
  --usr_iter=$usr_iter \
  --hash_size=$hash_size \
  --quick=$quick \
  --test-run=$test_run
done
