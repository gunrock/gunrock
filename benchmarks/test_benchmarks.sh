# ------------------------------------------------------------------------
# Algorithm benchmarking tests
# Run this from build directory
# If error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES: run with sudo
# Make sure to pass -DESSENTIALS_BUILD_BENCHMARKS=ON -DNVBench_ENABLE_CUPTI=ON to CMake
# ------------------------------------------------------------------------

#!/bin/bash

DATASET_DIR="../datasets"
BIN_DIR="./bin"

# Where to store output
JSON_DIR="json"

# Used for all algorithms except SPGEMM
MATRIX_FILE="${DATASET_DIR}/chesapeake/chesapeake.mtx"

# Used for Geo
COORDINATES_FILE="${DATASET_DIR}/geolocation/sample.labels"

# Used for SPGEMM
A_MATRIX="${DATASET_DIR}/spgemm/a.mtx"
B_MATRIX="${DATASET_DIR}/spgemm/b.mtx"

make bc_bench
make bfs_bench
make color_bench
make geo_bench
make hits_bench
make kcore_bench
make mst_bench
make ppr_bench
make pr_bench
make spgemm_bench
make spmv_bench
make sssp_bench
make tc_bench

${BIN_DIR}/bc_bench -m ${MATRIX_FILE}  --json ${JSON_DIR}/bc.json
${BIN_DIR}/bfs_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/bfs.json
${BIN_DIR}/color_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/color.json
${BIN_DIR}/geo_bench -m ${MATRIX_FILE} -c ${COORDINATES_FILE} --json ${JSON_DIR}/geo.json
${BIN_DIR}/hits_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/hits.json
${BIN_DIR}/kcore_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/kcore.json
${BIN_DIR}/mst_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/mst.json
${BIN_DIR}/ppr_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/ppr.json
${BIN_DIR}/pr_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/pr.json
${BIN_DIR}/spgemm_bench -a ${A_MATRIX} -b ${B_MATRIX} --json ${JSON_DIR}/spgemm.json
${BIN_DIR}/spmv_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/spmv.json
${BIN_DIR}/sssp_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/sssp.json
${BIN_DIR}/tc_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/tc.json
