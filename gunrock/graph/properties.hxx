#pragma once

namespace gunrock {
namespace graph {

/**
 * @brief define graph properties, this includes
 * is the graph directed? is the graph weighted?
 * Add more properties that we should support.
 * I can think of graph types (multigraph, cyclic, ...),
 * or negative edge support, etc.
 */
struct graph_properties_t {
  bool directed{false};
  bool weighted{false};
  graph_properties_t() = default;
};

}  // namespace graph
}  // namespace gunrock