#pragma once

namespace gunrock {
namespace graph {

// XXX: needs a better implementation
// maybe just a tuple.
template <typename vertex_t>
struct vertex_pair_t {
  vertex_t source;
  vertex_t destination;
};

template <typename edge_t>
struct edge_pair_t {
  edge_t x;
  edge_t y;
};

}  // namespace graph
}  // namespace gunrock