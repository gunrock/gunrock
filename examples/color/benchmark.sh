#!/bin/bash

# read -p "--JPL: " JPL
# read -p "--no-conflict: " no_conflict
# read -p "--seed: " seed
# read -p "--user-iter: " usr_iter
# read -p "--prohibit-size: " prohibit-size
# read -p "--graph-file: " graph_file
# read -p "--quick: " quick
# read -p "--test-run: " test_run

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

echo "[0] Run JPL with atomic"
echo "[1] Run JPL with user-defined iteration"
echo "[2] Run JPL atomic with min and max coloring"
echo "[3] Run JPL with min and max and user-defined iteration"
echo "[4] Run Hash implementation with atomic"
read -p "Option: " option

if [[ $option == "0" ]]
then
  logsFolder="logsJPLatomic"
  mkdir $logsFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    rm ./$logsFolder/$i.log
    touch ./$logsFolder/$i.log
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=true \
    --no-conflict=0 \
    --user-iter=0 \
    --prohibit-size=0 \
    --quick=true \
    --device=3 \
    --min-color=false \
    --test-run=true > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi

if [[ $option == "1" ]]
then
  echo "Specify number of iteration per line, Ctrl+D when done"
  while read line
  do
      ITR_ARRAY=("${ITR_ARRAY[@]}" $line)
  done

  logsFolder="logsJPLfast"
  mkdir $logsFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    rm ./$logsFolder/$i.log
    touch ./$logsFolder/$i.log
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=true \
    --no-conflict=0 \
    --user-iter=${ITR_ARRAY[$i]} \
    --prohibit-size=0 \
    --quick=true \
    --device=3 \
    --min-color=false \
    --test-run=false > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi


if [[ $option == "2" ]]
then
  logsFolder="logsJPLminmax"
  mkdir $logsFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    rm ./$logsFolder/$i.log
    touch ./$logsFolder/$i.log
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=true \
    --no-conflict=0 \
    --user-iter=0 \
    --prohibit-size=0 \
    --quick=true \
    --device=3 \
    --min-color=true \
    --test-run=true > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi


if [[ $option == "3" ]]
then
  echo "Specify number of iteration per line, Ctrl+D when done"
  while read line
  do
      ITR_ARRAY=("${ITR_ARRAY[@]}" $line)
  done
  logsFolder="logsJPLminmaxFast"
  mkdir $logsFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    rm ./$logsFolder/$i.log
    touch ./$logsFolder/$i.log
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=true \
    --no-conflict=0 \
    --user-iter=${ITR_ARRAY[$i]} \
    --prohibit-size=0 \
    --quick=true \
    --device=3 \
    --min-color=true \
    --test-run=false > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi



if [[ $option == "4" ]]
then
  logsFolder="logsHashAtomic"
  mkdir $logsFolder
  for i in `seq 0 11`;
  do
    echo "=================================================="
    echo "Testing Dataset $i: ${ADDR_ARRAY[$i]}"
    rm ./$logsFolder/$i.log
    touch ./$logsFolder/$i.log
    ./bin/test_color_10.0_x86_64 --graph-type=market \
    --graph-file=${ADDR_ARRAY[$i]} \
    --JPL=false \
    --no-conflict=1 \
    --user-iter=0 \
    --prohibit-size=0 \
    --quick=false \
    --device=3 \
    --undirected \
    --test-run=true > ./$logsFolder/$i.log
    grep -F "Max iteration" ./$logsFolder/$i.log
    grep -F "Number of colors" ./$logsFolder/$i.log
    grep -F "avg. elapsed" ./$logsFolder/$i.log
    echo "=================================================="
    printf "\n"
  done
fi
