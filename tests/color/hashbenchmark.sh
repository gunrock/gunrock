#!/bin/bash
#addresses
cyangDataSet="/data-2/topc-datasets/gc-data/"
addr0=${cyangDataSet}"offshore/offshore.mtx"
addr1=${cyangDataSet}"af_shell3/af_shell3.mtx"
addr2=${cyangDataSet}"parabolic_fem/parabolic_fem.mtx"
addr3=${cyangDataSet}"apache2/apache2.mtx"
addr4=${cyangDataSet}"ecology2/ecology2.mtx"
addr5=${cyangDataSet}"thermal2/thermal2.mtx"
addr6=${cyangDataSet}"G3_circuit/G3_circuit.mtx"
addr7=${cyangDataSet}"FEM_3D_thermal2/FEM_3D_thermal2.mtx"
addr8=${cyangDataSet}"thermomech_dK/thermomech_dK.mtx"
addr9=${cyangDataSet}"ASIC_320ks/ASIC_320ks.mtx"
addr10=${cyangDataSet}"cage13/cage13.mtx"
addr11=${cyangDataSet}"atmosmodd/atmosmodd.mtx"

ADDR_ARRAY=($addr0 $addr1 $addr2 $addr3 $addr4 $addr5 $addr6 $addr7 $addr8 $addr9 $addr10 $addr11)

echo "Enter max hash size"
read -p "Option: " max
echo "What dataset [0-11]"
read -p "Dataset: " dataset

logsFolder="hashResult"
mkdir $logsFolder
for i in `seq 0 $max`;
do
  echo "=================================================="
  echo "Testing Dataset $dataset: ${ADDR_ARRAY[$dataset]}"
  rm ./$logsFolder/$dataset.log
  touch ./$logsFolder/$dataset.log
  ./bin/test_color_10.0_x86_64 --graph-type=market \
  --graph-file=${ADDR_ARRAY[$dataset]} \
  --JPL=false \
  --user-iter=0 \
  --prohibit-size=$i \
  --quick=true \
  --device=3 \
  --undirected \
  --seed=123 \
  --test-run=true > ./$logsFolder/$i.log
  grep -F "Max iteration" ./$logsFolder/$i.log
  grep -F "Number of colors" ./$logsFolder/$i.log
  grep -F "avg. elapsed" ./$logsFolder/$i.log
  echo "=================================================="
  printf "\n"
done


