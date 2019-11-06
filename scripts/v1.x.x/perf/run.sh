# ------------------------------------------------------------------------
#  Gunrock (v1.x.x): Performance Testing Script(s)
# ------------------------------------------------------------------------

#!/bin/bash

TIMESTAMP=`date '+%Y-%m-%d_%H:%M:%S'`
EXEDIR="../../../build/bin"
DATADIR="/data/gunrock_dataset/large"
DEVICE="2"
TAG="Gunrock-v1-0-0_TitanXp"

for algo in bc; do # sssp bc pr cc; do
    ./${algo}.sh "$EXEDIR" "$DATADIR" "$DEVICE" "$TAG"
done
