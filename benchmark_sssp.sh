#!/bin/bash

# Benchmark SSSP with different load balancing strategies and algorithms

DATASETS=(
    "datasets/chesapeake/chesapeake.mtx"
    "datasets/bips98_606/bips98_606.mtx"
    "datasets/delaunay_n13/delaunay_n13.mtx"
    "datasets/delaunay_n24/delaunay_n24.mtx"
    "datasets/hollywood-2009/hollywood-2009.mtx"
    "datasets/road_usa/road_usa.mtx"
    "datasets/soc-LiveJournal1/soc-LiveJournal1.mtx"
)

LOAD_BALANCERS=("block_mapped" "merge_path")
ALGORITHMS=("DAWN" "classic")
NUM_RUNS=3

echo "=============================================="
echo "SSSP Benchmark Results"
echo "=============================================="
printf "%-30s %-15s %-12s " "Dataset" "Algorithm" "LoadBalance"
for i in $(seq 1 $NUM_RUNS); do
    printf "Run%d(ms) " $i
done
printf "Avg(ms)\n"
echo "----------------------------------------------"

for dataset in "${DATASETS[@]}"; do
    if [ ! -f "$dataset" ]; then
        continue
    fi
    
    name=$(basename "$dataset" .mtx)
    
    for alg in "${ALGORITHMS[@]}"; do
        for lb in "${LOAD_BALANCERS[@]}"; do
            times=()
            
            for run in $(seq 1 $NUM_RUNS); do
                if [ "$alg" == "DAWN" ]; then
                    result=$(./build_release/bin/sssp --market "$dataset" --src 0 --advance_load_balance "$lb" 2>&1 | grep "GPU Elapsed" | awk '{print $5}')
                else
                    result=$(./build_release/bin/sssp --market "$dataset" --src 0 --advance_load_balance "$lb" --algorithm classic 2>&1 | grep "GPU Elapsed" | awk '{print $5}')
                fi
                times+=($result)
            done
            
            # Calculate average
            avg=$(echo "${times[@]}" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; print sum/NF}')
            
            printf "%-30s %-15s %-12s " "$name" "$alg" "$lb"
            for t in "${times[@]}"; do
                printf "%-8.2f " $t
            done
            printf "%-8.2f\n" $avg
        done
    done
done
