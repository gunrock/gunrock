#!/bin/bash

# Validation script to ensure merge_path and block_mapped produce identical results

set -e

BUILD_DIR="${BUILD_DIR:-build_amd}"
BIN_DIR="${BUILD_DIR}/bin"

# Test datasets (use smaller ones for validation)
declare -a TEST_DATASETS=(
    "datasets/chesapeake/chesapeake.mtx"
)

declare -a ALGORITHMS=("sssp" "bfs")

echo "Validating merge_path vs block_mapped correctness..."
echo ""

ERRORS=0

for dataset in "${TEST_DATASETS[@]}"; do
    if [[ ! -f "${dataset}" ]]; then
        echo "Warning: Dataset ${dataset} not found, skipping..."
        continue
    fi
    
    dataset_name=$(basename "${dataset}" .mtx)
    echo "Testing dataset: ${dataset_name}"
    
    for algorithm in "${ALGORITHMS[@]}"; do
        if [[ ! -f "${BIN_DIR}/${algorithm}" ]]; then
            echo "Warning: Binary ${BIN_DIR}/${algorithm} not found, skipping..."
            continue
        fi
        
        echo "  Algorithm: ${algorithm}"
        
        # Test with source 0
        source=0
        
        # Run merge_path and capture output (distances)
        echo "    Running merge_path..."
        output_mp=$("${BIN_DIR}/${algorithm}" --market "${dataset}" \
            --advance_load_balance merge_path --src "${source}" 2>&1)
        time_mp=$(echo "$output_mp" | grep -oP 'GPU Elapsed Time\s*:\s*\K[0-9.]+' | head -1)
        
        # Extract distances (first 40 values)
        dist_mp=$(echo "$output_mp" | grep -oP 'GPU distances\[:40\]\s*=\s*\K[^\n]+' || echo "")
        
        # Run block_mapped and capture output
        echo "    Running block_mapped..."
        output_bm=$("${BIN_DIR}/${algorithm}" --market "${dataset}" \
            --advance_load_balance block_mapped --src "${source}" 2>&1)
        time_bm=$(echo "$output_bm" | grep -oP 'GPU Elapsed Time\s*:\s*\K[0-9.]+' | head -1)
        
        # Extract distances
        dist_bm=$(echo "$output_bm" | grep -oP 'GPU distances\[:40\]\s*=\s*\K[^\n]+' || echo "")
        
        # Compare distances
        if [[ "$dist_mp" == "$dist_bm" ]]; then
            echo "    ✓ Results match!"
            echo "      merge_path time: ${time_mp} ms"
            echo "      block_mapped time: ${time_bm} ms"
        else
            echo "    ✗ ERROR: Results do not match!"
            echo "      merge_path: ${dist_mp}"
            echo "      block_mapped: ${dist_bm}"
            ERRORS=$((ERRORS + 1))
        fi
        echo ""
    done
done

if [[ $ERRORS -eq 0 ]]; then
    echo "✓ All validations passed!"
    exit 0
else
    echo "✗ Validation failed with ${ERRORS} error(s)!"
    exit 1
fi
