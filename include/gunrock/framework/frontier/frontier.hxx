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
#include <gunrock/util/type_limits.hxx>

#include <gunrock/graph/graph.hxx>
#include <gunrock/cuda/context.hxx>

namespace gunrock {

using namespace memory;

// Maybe we use for frontier related function
namespace frontier {}  // namespace frontier

/**
 * @brief Underlying frontier data structure.
 */
enum frontier_storage_t {
  vector,
  bitmap,
  boolmap
};  // enum: frontier_storage_t

enum frontier_kind_t {
  edge_frontier,
  vertex_frontier
};  // enum: frontier_kind_t

template <typename t,
          frontier_storage_t underlying_st = frontier_storage_t::vector>
class frontier_t : public frontier::vector_frontier_t<t> {
 public:
  using type_t = t;
  using pointer_t = type_t*;
  using frontier_type_t = frontier_t<type_t>;

  // We can use std::conditional to figure out what type to use.
  using underlying_frontier_t = frontier::vector_frontier_t<type_t>;

  // <todo> revisit frontier constructors/destructor
  frontier_t()
      : underlying_frontier_t(),
        kind(frontier_kind_t::vertex_frontier),
        resizing_factor(1) {}
  frontier_t(std::size_t size, float frontier_resizing_factor = 1.0)
      : underlying_frontier_t(size),
        kind(frontier_kind_t::vertex_frontier),
        resizing_factor(frontier_resizing_factor) {}

  ~frontier_t() {}
  // </todo>

  /**
   * @brief Frontier type, either an edge based frontier or a vertex based
   * frontier.
   * @return frontier_kind_t
   */
  frontier_kind_t get_frontier_kind() const { return kind; }

  std::size_t get_size_in_bytes() const {
    return this->get_number_of_elements() * sizeof(type_t);
  }

  /**
   * @brief Get the number of elements within the frontier.
   * @return std::size_t
   */
  std::size_t get_number_of_elements() const {
    return underlying_frontier_t::get_number_of_elements();
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
  bool is_empty() const { return underlying_frontier_t::is_empty(); }

  /**
   * @brief (vertex-like) push back a value to the frontier.
   *
   * @param value
   */
  void push_back(type_t const& value) {
    underlying_frontier_t::push_back(value);
  }

  /**
   * @brief Fill the entire frontier with a user-specified value.
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
    sort::radix::sort_keys(this->data(), this->number_of_elements(), stream);
  }

  /**
   * @brief Print the frontier.
   */
  void print() { underlying_frontier_t::print(); }

 private:
  frontier_kind_t kind;   // vertex or edge frontier.
  float resizing_factor;  // reserve size * factor.
};                        // struct frontier_t

}  // namespace gunrock