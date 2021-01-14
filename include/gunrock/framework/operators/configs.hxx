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
  merge_path,      // Merrill & Garland (SpMV)
  bucketing,       // Davidson et al. (SSSP)
  work_stealing,   // <cite>
  input_oriented,  // Input-Oriented
  all_edges        // Output-Oriented (advance entire graph)
};

enum advance_type_t {
  vertex_to_vertex,
  vertex_to_edge,
  edge_to_edge,
  edge_to_vertex
};

enum advance_direction_t {
  forward,   // Push-based approach
  backward,  // Pull-based approach
  optimized  // Push-pull optimized
};

enum filter_algorithm_t {
  predicated,  // Copy if predicate = true
  compact,     // 2-Pass Transform compact
  bypass       // Marks as invalid, instead of culling
};

}  // namespace operators
}  // namespace gunrock