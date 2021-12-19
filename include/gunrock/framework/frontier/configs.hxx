/**
 * @file configs.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-12-16
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

namespace gunrock {
namespace frontier {

/**
 * @brief Underlying frontier data structure.
 */
enum frontier_view_t {
  vector,  /// vector-based frontier
  bitmap,  /// bitmap-based frontier
  boolmap  /// boolmap-based frontier
};         // enum: frontier_view_t

/**
 * @brief Type of frontier (vertex or edge)
 * @todo Use a better name than frontier_kind_t.
 */
enum frontier_kind_t {
  vertex_frontier,      /// vertex frontier storage only
  edge_frontier,        /// edge frontier storage only
  vertex_edge_frontier  /// (wip)
};                      // enum: frontier_kind_t

}  // namespace frontier
}  // namespace gunrock