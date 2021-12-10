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

#include <gunrock/framework/frontier/vector_frontier.hxx>
#include <gunrock/framework/frontier/boolmap_frontier.hxx>

#include <gunrock/util/type_limits.hxx>

#include <gunrock/graph/graph.hxx>
#include <gunrock/cuda/context.hxx>

namespace gunrock {

using namespace memory;

/**
 * @brief Underlying frontier data structure.
 */
enum frontier_storage_t {
  vector,
  bitmap,
  boolmap
};  // enum: frontier_storage_t

enum frontier_kind_t {
  edge_frontier,        /// edge frontier storage only
  vertex_frontier,      /// vertex frontier storage only
  vertex_edge_frontier  /// (wip)
};                      // enum: frontier_kind_t

template <typename t,
          frontier_storage_t underlying_st = frontier_storage_t::vector>
class frontier_t
    : public std::conditional_t<underlying_st == frontier_storage_t::vector,
                                frontier::vector_frontier_t<t>,
                                frontier::boolmap_frontier_t<t>> {
 public:
  using type_t = t;
  using pointer_t = type_t*;
  using frontier_type_t = frontier_t<type_t>;

  // We can use std::conditional to figure out what type to use.
  using underlying_frontier_t =
      std::conditional_t<underlying_st == frontier_storage_t::vector,
                         frontier::vector_frontier_t<type_t>,
                         frontier::boolmap_frontier_t<type_t>>;

  // Constructors
  frontier_t()
      : underlying_frontier_t(),
        kind(frontier_kind_t::vertex_frontier),
        resizing_factor(1) {}
  frontier_t(std::size_t size, float frontier_resizing_factor = 1.0)
      : underlying_frontier_t(size),
        kind(frontier_kind_t::vertex_frontier),
        resizing_factor(frontier_resizing_factor) {}

  // Empty Destructor, this is important on kernel-exit.
  ~frontier_t() {}

  // Copy Constructor
  frontier_t(const frontier_t& rhs)
      : underlying_frontier_t(rhs),
        kind(rhs.kind),
        resizing_factor(rhs.resizing_factor) {}

  // Disable move and assignment.
  frontier_t& operator=(const frontier_t& rhs) = delete;
  frontier_t& operator=(frontier_t&&) = delete;
  frontier_t(frontier_t&&) = delete;

  /**
   * @brief Frontier type, either an edge based frontier or a vertex based
   * frontier.
   * @return frontier_kind_t
   */
  frontier_kind_t get_frontier_kind() const { return kind; }

  constexpr frontier_storage_t get_frontier_storage_t() const {
    return underlying_st;
  }

  std::size_t get_size_in_bytes() {
    return this->get_number_of_elements() * sizeof(type_t);
  }

  /**
   * @brief Get the number of elements within the frontier.
   * @return std::size_t
   */
  std::size_t get_number_of_elements(cuda::stream_t stream = 0) {
    return underlying_frontier_t::get_number_of_elements(stream);
  }

  /**
   * @brief Get the resizing factor used to scale the frontier size.
   *
   * @return float
   */
  float get_resizing_factor() const { return resizing_factor; }

  /**
   * @brief Get the capacity (number of elements possible).
   * @return std::size_t
   */
  std::size_t get_capacity() const {
    return underlying_frontier_t::get_capacity();
  }

  /**
   * @brief Set the frontier kind: edge or vertex frontier.
   *
   * @param _kind
   */
  void set_frontier_kind(frontier_kind_t _kind) { kind = _kind; }

  /**
   * @brief Set the resizing factor for the frontier. This float is used to
   * multiply the `reserve` size to scale it a bit higher every time to avoid
   * reallocations.
   *
   * @param factor
   */
  void set_resizing_factor(float factor) { resizing_factor = factor; }

  /**
   * @brief Set how many number of elements the frontier contains. Note, this is
   * manually managed right now, we can look for better and cleaner options
   * later on as well. We require users (gunrock devs), to set the number of
   * elements after figuring it out.
   *
   * @param elements
   */
  void set_number_of_elements(std::size_t const& elements) {
    underlying_frontier_t::set_number_of_elements(elements);
  }

  pointer_t data() { return underlying_frontier_t::data(); }
  pointer_t begin() { return underlying_frontier_t::begin(); }
  pointer_t end() { return underlying_frontier_t::end(); }
  // bool is_empty() const { return underlying_frontier_t::is_empty(); }

  /**
   * @brief Fill the entire frontier with a user-specified value. For boolmap
   * frontier, only valid values are 1s or 0s.
   *
   * @param value
   * @param stream
   */
  void fill(type_t const value, cuda::stream_t stream = 0) {
    underlying_frontier_t::fill(value, stream);
  }

  /**
   * @brief Resize the underlying frontier storage to be exactly the size
   * specified. Note that this actually resizes, and will now change the
   * capacity as well as the size.
   *
   * @param size number of elements used to resize the frontier (count not
   * bytes).
   * @param default_value
   */
  // void resize(
  //     std::size_t const& size,
  //     type_t const default_value =
  //     gunrock::numeric_limits<type_t>::invalid()) {
  //   underlying_frontier_t::resize(size, default_value);
  //   this->set_number_of_elements(size);
  // }

  /**
   * @brief "Hints" the alocator that we need to reserve the suggested size. The
   * capacity() will increase and report reserved() size, but size() will still
   * report the actual size, not reserved size. See std::vector for more detail.
   *
   * @param size size to reserve (size is in count not bytes).
   */
  void reserve(std::size_t const& size) {
    underlying_frontier_t::reserve(size * resizing_factor);
  }

  /**
   * @brief Parallel sort the frontier.
   *
   * @param order see sort::order_t
   * @param stream see cuda::stream
   */
  void sort(sort::order_t order = sort::order_t::ascending,
            cuda::stream_t stream = 0) {
    underlying_frontier_t::sort(order, stream);
  }

  /**
   * @brief Print the frontier.
   */
  void print() { underlying_frontier_t::print(); }

 private:
  frontier_kind_t kind;   // vertex or edge frontier.
  float resizing_factor;  // reserve size * factor.
};                        // struct frontier_t

// Maybe we use for frontier related function
namespace frontier {

/**
 * @brief Get the element at the specified index.
 *
 * @tparam type_t
 * @param idx
 * @param ptr
 * @return type_t
 */
template <frontier_storage_t underlying_st = frontier_storage_t::vector,
          typename type_t>
__device__ __forceinline__ type_t get_element_at(std::size_t const& idx,
                                                 type_t* ptr) {
  auto element = thread::load(ptr + idx);
  if (underlying_st == frontier_storage_t::boolmap) {
    if (element == 1)
      return idx;
    else
      return gunrock::numeric_limits<type_t>::invalid();
  } else
    return element;
}

/**
 * @brief Set the element at the specified index.
 *
 * @tparam type_t
 * @param idx
 * @param element
 * @param ptr
 * @return void
 */
template <frontier_storage_t underlying_st = frontier_storage_t::vector,
          typename type_t>
__device__ __forceinline__ void set_element_at(std::size_t const& idx,
                                               type_t const& element,
                                               type_t* ptr) {
  if (underlying_st == frontier_storage_t::boolmap) {
    thread::store(ptr + idx, 1);
  } else
    thread::store(ptr + idx, element);
}
}  // namespace frontier
}  // namespace gunrock