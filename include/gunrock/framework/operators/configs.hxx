/**
 * @file configs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-10-20
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

namespace gunrock {
namespace operators {

/**
 * @brief Load balancing type options, not all operators support load-balancing
 * or have need to balance work.
 *
 */
enum load_balance_t {
  merge_path,     // Merrill & Garland (SpMV)
  bucketing,      // Davidson et al. (SSSP)
  work_stealing,  // <cite>
  thread_mapped,  // 1 element / thread
  warp_mapped,    // Equal # of elements / warp
  block_mapped,   // Equal # of elements / block
  all_edges       // 1 edge / thread (advance an entire graph)
};

enum advance_type_t {
  vertex_to_vertex,  // Vertex input to vertex output frontier
  vertex_to_edge,    // Vertex input to edge output frontier
  edge_to_edge,      // Edge input to edge output frontier
  edge_to_vertex     // Edge input to vertex output frontier
};

enum advance_direction_t {
  forward,   // Push-based approach
  backward,  // Pull-based approach
  optimized  // Push-pull optimized
};

enum filter_algorithm_t {
  remove,      // Remove if predicate = true
  predicated,  // Copy if predicate = true
  compact,     // 2-Pass Transform compact
  bypass       // Marks as invalid, instead of culling
};

}  // namespace operators
}  // namespace gunrock