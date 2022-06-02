# ------------------------------------------------------------------------
# Algorithm benchmarking tests
# Run this from build directory
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

sudo ${BIN_DIR}/bc_bench -m ${MATRIX_FILE}  --json ${JSON_DIR}/bc.json
sudo ${BIN_DIR}/bfs_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/bfs.json
sudo ${BIN_DIR}/color_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/color.json
sudo ${BIN_DIR}/geo_bench -m ${MATRIX_FILE} -c ${COORDINATES_FILE} --json ${JSON_DIR}/geo.json
sudo ${BIN_DIR}/hits_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/hits.json
sudo ${BIN_DIR}/kcore_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/kcore.json
sudo ${BIN_DIR}/mst_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/mst.json
sudo ${BIN_DIR}/ppr_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/ppr.json
sudo ${BIN_DIR}/pr_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/pr.json
sudo ${BIN_DIR}/spgemm_bench -a ${A_MATRIX} -b ${B_MATRIX} --json ${JSON_DIR}/spgemm.json
sudo ${BIN_DIR}/spmv_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/spmv.json
sudo ${BIN_DIR}/sssp_bench -m ${MATRIX_FILE} --json ${JSON_DIR}/sssp.json
