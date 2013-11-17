// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * dobfs_functor.cuh
 *
 * @brief Device functions for DOBFS problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bfs/dobfs_problem.cuh>

namespace gunrock {
namespace app {
namespace dobfs {

// TODO:
// 
// Prepare for reverse BFS (first two functor set)
// 1) prepare unvisited queue
//   VertexMap for all nodes, select whose label is -1
// 2) prepare frontier_map_in
//   Use MemsetKernel to set all frontier_map_in as 0
//   Vertexmap for all the nodes in current frontier,
//   set their frontier_map_in value as 1
// 3) clear all frontier_map_in value as 0
//
// During the reverse BFS (third functor set)
// 1) BackwardEdgeMap
// 2) Clear frontier_map_in
// 3) VertexMap
//
// Switch back to normal BFS (final functor set)
// 1) prepare current frontier
// VertexMap for all nodes, select whose frontier_map_out is 1
//

} // dobfs
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
