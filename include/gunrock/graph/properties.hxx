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
  bool weighted{true};
  graph_properties_t() = default;
};

enum view_t : uint32_t {
  csr = 1 << 1,
  csc = 1 << 2,
  coo = 1 << 3,
  invalid = 1 << 0
};

constexpr inline view_t operator|(view_t lhs, view_t rhs) {
  return static_cast<view_t>(static_cast<uint32_t>(lhs) |
                             static_cast<uint32_t>(rhs));
}

constexpr inline view_t set(view_t lhs, view_t rhs) {
  return static_cast<view_t>(static_cast<uint32_t>(lhs) |
                             static_cast<uint32_t>(rhs));
}

constexpr inline view_t unset(view_t lhs, view_t rhs) {
  return static_cast<view_t>(static_cast<uint32_t>(lhs) &
                             ~static_cast<uint32_t>(rhs));
}

constexpr inline bool has(view_t lhs, view_t rhs) {
  return (static_cast<uint32_t>(lhs) & static_cast<uint32_t>(rhs)) ==
         static_cast<uint32_t>(rhs);
}

constexpr inline view_t toggle(view_t lhs, view_t rhs) {
  return static_cast<view_t>(static_cast<uint32_t>(lhs) ^
                             static_cast<uint32_t>(rhs));
}

}  // namespace graph
}  // namespace gunrock