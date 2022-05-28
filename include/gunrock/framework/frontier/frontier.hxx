/**
 * @file frontier.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Frontier structure for graph algorithms. Frontiers are one of the
 * fundamental structure used to implement graph algorithms on the GPU for
 * gunrock. The concept simply implies that we have either vertex or edge
 * frontiers, and all operations are applied on the frontier rather than the
 * graph itself. These operations (operators) are parallel, and exposes the
 * data-centric graph abstraction within gunrock.
 *
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <gunrock/framework/frontier/configs.hxx>
#include <gunrock/framework/frontier/vector_frontier.hxx>
// #include <gunrock/framework/frontier/experimental/boolmap_frontier.hxx>

#include <gunrock/util/type_limits.hxx>

#include <gunrock/graph/graph.hxx>
#include <gunrock/cuda/context.hxx>

namespace gunrock {
namespace frontier {
using namespace memory;

template <typename vertex_t,
          typename edge_t,
          frontier_kind_t _kind = frontier_kind_t::vertex_frontier,
          frontier_view_t _view = frontier_view_t::vector>
class frontier_t : public frontier::vector_frontier_t<vertex_t, edge_t, _kind> {
 public:
  using vertex_type = vertex_t;
  using edge_type = edge_t;
  using type_t = std::conditional_t<_kind == frontier_kind_t::vertex_frontier,
                                    vertex_t,
                                    edge_t>;
  using offset_t = std::conditional_t<_kind == frontier_kind_t::vertex_frontier,
                                      edge_t,
                                      vertex_t>;
  using frontier_type = frontier_t<vertex_t, edge_t, _kind, _view>;

  /// TODO: This is a more permenant solution.
  // We can use std::conditional to figure out what type to use.
  // using underlying_view_t = std::conditional_t<
  //     _view == frontier_view_t::vector,
  //     frontier::vector_frontier_t<vertex_t, edge_t, _kind>,
  //     experimental::frontier::boolmap_frontier_t<vertex_t, edge_t, _kind>>;

  using underlying_view_t =
      frontier::vector_frontier_t<vertex_t, edge_t, _kind>;

  /**
   * @brief Default constructor.
   */
  frontier_t() : underlying_view_t() {}

  /**
   * @brief Construct a new frontier_t object, this constructor is only
   * available with frontier_view_t == vector.
   * @ref
   * https://stackoverflow.com/questions/17842478/select-class-constructor-using-enable-if
   * @tparam U
   * @param size
   * @param frontier_resizing_factor
   */
  template <typename U = underlying_view_t>
  frontier_t(
      std::size_t size,
      float frontier_resizing_factor = 1.0,
      typename std::enable_if<std::is_same<
          U,
          frontier::vector_frontier_t<vertex_t, edge_t, _kind>>::value>::type* =
          nullptr)
      : underlying_view_t(size, frontier_resizing_factor) {}

  /**
   * @brief Destroy the frontier t object. This is an empty destructor, which is
   * important such that when a kernel call exits, the frontier is not
   * destroyed.
   *
   */
  ~frontier_t() {}

  /**
   * @brief Custom copy constructor.
   * @param rhs
   */
  __device__ __host__ frontier_t(const frontier_t& rhs)
      : underlying_view_t(rhs) {}

  /**
   * @brief Returns what the frontier is storing, edges or vertices.
   * @return frontier_kind_t
   */
  __host__ __device__ __forceinline__ constexpr frontier_kind_t get_kind()
      const {
    return kind;
  }

  /**
   * @brief Returns the underlying view of the storage of the frontier.
   * @return frontier_view_t
   */
  __host__ __device__ __forceinline__ constexpr frontier_view_t get_view()
      const {
    return view;
  }

  /**
   * @brief Get the number of elements within the frontier.
   * @return std::size_t
   */
  __host__ __device__ __forceinline__ std::size_t get_number_of_elements(
      gcuda::stream_t stream = 0) {
    return underlying_view_t::get_number_of_elements(stream);
  }

  /**
   * @brief Is the frontier empty or not? For certain underlying data
   * structures, this is an expensive operation (for example, boolmap) if the
   * number of elements in a frontier is not self-managed. Meaning, if the user
   * is not keeping track of the number of elements within a frontier.
   *
   * @return true if the frontier is empty.
   * @return false if the frontier is not empty.
   */
  bool is_empty() const { return underlying_view_t::is_empty(); }

  /**
   * @brief Print the frontier, for a vector frontier it simply prints the
   * elements. For a boolmap it prints the indices where the data is true.
   * @todo Add support for printing the frontier in a more readable format.
   * @todo Add support for printing a specific range of elements.
   */
  void print() { underlying_view_t::print(); }

 private:
  static constexpr frontier_view_t view = _view;  // vector, boolmap, etc...
  static constexpr frontier_kind_t kind = _kind;  // vertex or edge frontier.

};  // struct frontier_t

}  // namespace frontier
}  // namespace gunrock