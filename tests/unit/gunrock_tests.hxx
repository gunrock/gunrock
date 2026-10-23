/**
 * @file unittests.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief All unit tests includes for the gunrock library.
 * @date 2021-12-24
 *
 * @copyright Copyright (c) 2021
 *
 */

#include "formats/format.cuh"

#include "graph/graph.cuh"

#include "memory/memory.cuh"

#include "framework/operators/for.cuh"
#include "framework/operators/advance/merge_path.cuh"

#include "utils/type_limits.cuh"

#include "cuda/launch_box.cuh"
#include "cuda/context.cuh"
#include "cuda/device_properties.cuh"

#include "containers/array.cuh"

#include "io/matrix_market.cuh"
#include "io/smtx.cuh"

#include "algorithms/bfs.cuh"
#include "algorithms/bc.cuh"
#include "algorithms/tc.cuh"
