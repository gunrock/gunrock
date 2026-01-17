#!/bin/bash

# Benchmark script to compare merge_path vs block_mapped load balancing
# Usage: ./benchmark_merge_path.sh [output_csv_file]

set -e

# Configuration
NUM_RUNS=${NUM_RUNS:-5}  # Number of runs per configuration for statistical significance (default: 5)
WARMUP_RUNS=${WARMUP_RUNS:-1}  # Warm-up runs before timing (default: 1)
BUILD_DIR="${BUILD_DIR:-build_amd}"
BIN_DIR="${BUILD_DIR}/bin"
RESULTS_DIR="benchmarks/results"
PLOTS_DIR="benchmarks/plots"

# Create output directories
mkdir -p "${RESULTS_DIR}"
mkdir -p "${PLOTS_DIR}"

# Output CSV file
OUTPUT_CSV="${1:-${RESULTS_DIR}/merge_path_comparison.csv}"

# Datasets to benchmark
declare -a DATASETS=(
    "datasets/hollywood-2009/hollywood-2009.mtx"
    "datasets/delaunay_n24/delaunay_n24.mtx"
    "datasets/arabic-2005/arabic-2005.mtx"
    "datasets/road_usa/road_usa.mtx"
)

# Algorithms to test
declare -a ALGORITHMS=("sssp" "bfs")

# Load balancing methods
declare -a LB_METHODS=("merge_path" "block_mapped")

# Source nodes to test (0, and some random ones - we'll generate random ones per dataset)
# We'll use source 0 and generate a few random sources based on graph size

echo "Starting benchmark: merge_path vs block_mapped"
echo "Output file: ${OUTPUT_CSV}"
echo "Number of runs per configuration: ${NUM_RUNS}"
echo ""

# Write CSV header
echo "dataset,algorithm,load_balance,source_node,run_number,time_ms,vertices,edges" > "${OUTPUT_CSV}"

# Function to extract graph statistics from matrix market file
get_graph_stats() {
    local dataset="$1"
    # Matrix market format: first non-comment line has: rows cols edges
    if [[ -f "${dataset}" ]]; then
        local header=$(grep -v "^%" "${dataset}" | head -1)
        local vertices=$(echo "$header" | awk '{print $1}')
        local edges=$(echo "$header" | awk '{print $3}')
        echo "${vertices},${edges}"
    else
        echo "unknown,unknown"
    fi
}

# Function to run benchmark and extract timing
run_benchmark() {
    local dataset="$1"
    local algorithm="$2"
    local lb_method="$3"
    local source="$4"
    local run_num="$5"
    
    # Run with timeout to handle potential hangs
    local output=$(timeout 120 "${BIN_DIR}/${algorithm}" \
        --market "${dataset}" \
        --advance_load_balance "${lb_method}" \
        --src "${source}" \
        2>&1 || echo "ERROR")
    
    if [[ "$output" == *"ERROR"* ]] || [[ "$output" == *"error"* ]]; then
        echo "ERROR" >&2
        return 1
    fi
    
    # Extract GPU Elapsed Time
    local time_ms=$(echo "$output" | grep -oP 'GPU Elapsed Time\s*:\s*\K[0-9.]+' | head -1 || echo "")
    
    if [[ -z "$time_ms" ]]; then
        echo "ERROR" >&2
        return 1
    fi
    
    echo "$time_ms"
    return 0
}

# Function to get random source nodes
get_random_sources() {
    local dataset="$1"
    local num_sources=3
    
    # Get graph size from matrix market file (first non-comment line)
    if [[ -f "${dataset}" ]]; then
        local header=$(grep -v "^%" "${dataset}" | head -1)
        local num_vertices=$(echo "$header" | awk '{print $1}')
    else
        local num_vertices=1000
    fi
    
    # Generate random sources (include 0 and a few random ones)
    local sources="0"
    for i in $(seq 1 $num_sources); do
        local rand_source=$((RANDOM % num_vertices))
        sources="${sources} ${rand_source}"
    done
    
    echo "$sources"
}

# Main benchmarking loop
for dataset in "${DATASETS[@]}"; do
    if [[ ! -f "${dataset}" ]]; then
        echo "Warning: Dataset ${dataset} not found, skipping..."
        continue
    fi
    
    dataset_name=$(basename "${dataset}" .mtx)
    echo "Processing dataset: ${dataset_name}"
    
    # Get graph statistics
    stats=$(get_graph_stats "${dataset}")
    vertices=$(echo "$stats" | cut -d',' -f1)
    edges=$(echo "$stats" | cut -d',' -f2)
    
    # Get source nodes to test
    sources=$(get_random_sources "${dataset}")
    
    for algorithm in "${ALGORITHMS[@]}"; do
        if [[ ! -f "${BIN_DIR}/${algorithm}" ]]; then
            echo "Warning: Binary ${BIN_DIR}/${algorithm} not found, skipping..."
            continue
        fi
        
        for lb_method in "${LB_METHODS[@]}"; do
            for source in $sources; do
                echo "  Testing: ${algorithm} with ${lb_method}, source=${source}"
                
                # Warm-up runs
                for warmup in $(seq 1 ${WARMUP_RUNS}); do
                    run_benchmark "${dataset}" "${algorithm}" "${lb_method}" "${source}" "warmup" > /dev/null 2>&1 || true
                done
                
                # Actual benchmark runs
                for run in $(seq 1 ${NUM_RUNS}); do
                    time_ms=$(run_benchmark "${dataset}" "${algorithm}" "${lb_method}" "${source}" "${run}")
                    
                    if [[ "$time_ms" != "ERROR" ]] && [[ -n "$time_ms" ]]; then
                        echo "${dataset_name},${algorithm},${lb_method},${source},${run},${time_ms},${vertices},${edges}" >> "${OUTPUT_CSV}"
                        echo "    Run ${run}: ${time_ms} ms"
                    else
                        echo "    Run ${run}: FAILED" >&2
                    fi
                done
            done
        done
    done
    echo ""
done

echo "Benchmarking complete!"
echo "Results saved to: ${OUTPUT_CSV}"
echo "Total lines: $(wc -l < "${OUTPUT_CSV}")"
