#!/bin/bash

# get all execution files in ./bin
files=./bin/*
# split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
exe_file=${arr[0]}
echo $exe_file market  ../../dataset/small/chesapeake.mtx --pattern-graph-type=market --pattern-graph-file=../../dataset/small/query_sm.mtx --undirected=1 --pattern-undirected=1 --num-runs=1 --quick=True
$exe_file market  ../../dataset/small/chesapeake.mtx --pattern-graph-type=market --pattern-graph-file=../../dataset/small/query_sm.mtx --undirected=1 --pattern-undirected=1 --num-runs=1 --quick=True

