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

echo "[0] Run JPL"
echo "[1] Run Hash"
echo "[2] Run Advance Reduce"
read -p "Option: " option

if [[ $option == "0" ]]
then
  echo "Specify number of iteration per line, Ctrl+D when done"
  while read line
  do
      ITR_ARRAY=("${ITR_ARRAY[@]}" $line)
  done
  logsFolder="logsJPL"
  jsonFolder="jsonJPL"
  rm -rf $logsFolder
  rm -rf $jsonFolder
  mkdir $logsFolder
  mkdir $jsonFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=true \
    --LBCOLOR=false \
    --seed=12312 \
    --undirected \
    --no-conflict=0 \
    --user-iter=${ITR_ARRAY[$i]} \
    --prohibit-size=0 \
    --jsondir="/home/mstruong/gunrock/tests/color/$jsonFolder" \
    --tag="K40-JPL-GRAPL" \
    --quick=false \
    --test-run=false > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi


if [[ $option == "1" ]]
then
  echo "what is hash size?"
  read -p "Size: " size
  logsFolder="logsHash-$size"
  jsonFolder="jsonHash-$size"
  rm -rf $logsFolder
  rm -rf $jsonFolder
  mkdir $logsFolder
  mkdir $jsonFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=false \
    --LBCOLOR=false \
    --seed=12312 \
    --undirected \
    --no-conflict=1 \
    --user-iter=0 \
    --prohibit-size=$size \
    --jsondir="/home/mstruong/gunrock/tests/color/$jsonFolder" \
    --tag="K40-HASH-GRAPL" \
    --quick=false \
    --test-run=true > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi

if [[ $option == "2" ]]
then
  echo "Specify number of iteration per line, Ctrl+D when done"
  while read line
  do
      ITR_ARRAY=("${ITR_ARRAY[@]}" $line)
  done
  logsFolder="logsAR"
  jsonFolder="jsonAR"
  rm -rf $logsFolder
  rm -rf $jsonFolder
  mkdir $logsFolder
  mkdir $jsonFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=true \
    --LBCOLOR=true \
    --seed=12312 \
    --undirected \
    --no-conflict=0 \
    --user-iter=${ITR_ARRAY[$i]} \
    --prohibit-size=0 \
    --jsondir="/home/mstruong/gunrock/tests/color/$jsonFolder" \
    --tag="K40-AR-GRAPL" \
    --quick=false \
    --test-run=true > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi
