#pragma once

#include <gunrock/graph/graph.hxx>

namespace gunrock {

template <typename graph_type>
struct problem_t {
  std::shared_ptr<graph_type> graph_slice;

  problem_t() : graph_slice(std::make_shared<graph_type>()) {}

  // Disable copy ctor and assignment operator.
  // We do not want to let user copy only a slice.
  // Explanation:
  // https://www.geeksforgeeks.org/preventing-object-copy-in-cpp-3-different-ways/
  problem_t(const problem_t& rhs) = delete;             // Copy constructor
  problem_t& operator=(const problem_t& rhs) = delete;  // Copy assignment

  problem_t(std::shared_ptr<graph_type> rhs) { graph_slice = rhs; }

};  // struct problem_t

}  // namespace gunrock