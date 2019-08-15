#!/bin/bash

# get all execution files in ./bin
files=./bin/*
# split file names into arr
arr=$(echo $files | tr " " "\n")
max_ver_num="$"
exe_file=${arr[0]}
echo $exe_file market  ../../dataset/small/tree.mtx --pattern-graph-type=market --pattern-graph-file=../../dataset/small/query_sm.mtx --undirected=1 --pattern-undirected=1 --num-runs=1 --quiet=False
$exe_file market  ../../dataset/small/tree.mtx --pattern-graph-type=market --pattern-graph-file=../../dataset/small/query_sm.mtx --undirected=1 --pattern-undirected=1 --num-runs=1 --quiet=False

