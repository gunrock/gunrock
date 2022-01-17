#pragma once

namespace gunrock {
namespace graph {

/**
 * @brief vertex pair of source and destination, accessed using .source and
 * .destination;
 *
 * @tparam vertex_t vertex type.
 */
template <typename vertex_t>
struct alignas(8) vertex_pair_t {
  vertex_t source;
  vertex_t destination;
};

/// @todo: Cannot remember what this is for.
template <typename edge_t>
struct edge_pair_t {
  edge_t x;
  edge_t y;
};

}  // namespace graph
}  // namespace gunrock