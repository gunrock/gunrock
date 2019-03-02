#!/bin/bash

TIMESTAMP=`date '+%Y-%m-%d_%H:%M:%S'`

EXEDIR=${1:-"../../build/bin"}
DATADIR=${2:-"/data/gunrock_dataset/large"}
DEVICE=${3:-"0"}
TAG=${4:-"$TIMESTAMP"}

for algo in bfs sssp bc pr cc; do
    ./${algo}-test.sh "$EXEDIR" "$DATADIR" "$DEVICE" "$TAG"
done
