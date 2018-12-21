#!/bin/bash

read -p "--JPL: " JPL
read -p "--no-conflict: " no_conflict
read -p "--seed: " seed
read -p "--user-iter: " usr_iter
read -p "--hash-size: " hash_size
read -p "--graph-file: " graph_file
read -p "--quick: " quick
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

ADDR_ARRAY=(
  $addr1 $addr2 $addr3 $addr4 $addr5 $addr6
  $addr7 $addr8 $addr9 $addr10 $addr11 $addr12
)

for i in `seq 1 12`;
do
  echo "/data-2/topc-datasets/gc-data/${ADDR_ARRAY[$i]}"
  rm ./logs/$i.log
  touch ./logs/$i.log
  ./bin/test_color_10.0_x86_64 --graph-type=market \
  --graph-file=/data-2/topc-datasets/gc-data/${ADDR_ARRAY[$i]} \
  --JPL=$JPL \
  --no-conflict=$no_conflict \
  --seed=$seed \
  --user-iter=$usr_iter \
  --hash-size=$hash_size \
  --quick=$quick \
  --test-run=$test_run > ./logs/$i.log
done

