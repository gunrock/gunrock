/**
 * @file boolmap_frontier.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Boolmap-based frontier implementation.
 * @version 0.1
 * @date 2021-03-12
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/util/type_limits.hxx>
#include <gunrock/container/vector.hxx>
#include <gunrock/algorithms/sort/radix_sort.hxx>
#include <thrust/sequence.h>

namespace gunrock {
namespace frontier {
using namespace memory;

template <typename type_t>
class boolmap_frontier_t {
 public:
  using pointer_t = type_t*;

  // Constructors
  boolmap_frontier_t() : storage(), num_elements(0) {}
  boolmap_frontier_t(std::size_t size) : storage(size), num_elements(size) {}

  // Copy Constructor
  template <typename frontier_t_t>
  boolmap_frontier_t(const frontier_t_t& other)
      : storage(other.storage), num_elements(other.num_elements) {}

  /**
   * @brief Get the number of elements within the frontier. This is a costly
   * feature of a boolmap and should be avoided when possible.
   * @return std::size_t
   */
  std::size_t get_number_of_elements(cuda::stream_t stream = 0) {
    // Compute number of elements using a reduction.
    num_elements = thrust::reduce(thrust::cuda::par.on(stream), this->begin(),
                                  this->end(), 0);
    return num_elements;
  }

  /**
   * @brief Get the capacity (number of elements possible).
   * @return std::size_t
   */
  std::size_t get_capacity() const { return storage.capacity(); }

  /**
   * @brief Set how many number of elements the frontier contains. Note, this is
   * manually managed right now, we can look for better and cleaner options
   * later on as well. We require users (gunrock devs), to set the number of
   * elements after figuring it out.
   *
   * @param elements
   */
  void set_number_of_elements(std::size_t const& elements) {
    num_elements = elements;
  }

  pointer_t data() { return raw_pointer_cast(storage.data()) /* .get() */; }
  pointer_t begin() { return this->data(); }
  pointer_t end() { return this->begin() + this->get_number_of_elements(); }

  /**
   * @brief Is the frontier empty or not?
   * @todo right now, this relies on an expensive get_number_of_elements() call,
   * we can replace this with a simple transform that checks each position and
   * if any one of the position is active, the frontier is not empty.
   *
   * @return true
   * @return false
   */
  bool is_empty() { return (this->get_number_of_elements() == 0); }

  /**
   * @brief Fill the entire frontier with a user-specified value.
   *
   * @param value
   * @param stream
   */
  void fill(type_t const value, cuda::stream_t stream = 0) {
    if (value != 0 || value != 1)
      error::throw_if_exception(cudaErrorUnknown,
                                "Boolmap only supports 1 or 0 as fill value.");

    thrust::fill(thrust::cuda::par.on(stream), this->begin(), this->end(),
                 value);
  }

  /**
   * @brief Resize the underlying frontier storage to be exactly the size
   * specified. Note that this actually resizes, and will now change the
   * capacity as well as the size.
   *
   * @param size number of elements used to resize the frontier (count not
   * bytes).
   * @param default_value is 0 (meaning vertex is not active).
   */
  void resize(std::size_t const& size, type_t const default_value = 0) {
    storage.resize(size, default_value);
  }

  /**
   * @brief "Hints" the alocator that we need to reserve the suggested size. The
   * capacity() will increase and report reserved() size, but size() will still
   * report the actual size, not reserved size.
   * @note This isn't as relevant for a boolmap because the size of the frontier
   * remains the same.
   *
   * @param size size to reserve (size is in count not bytes).
   */
  void reserve(std::size_t const& size) { storage.reserve(size); }

  /**
   * @brief Parallel sort the frontier.
   *
   * @param order see sort::order_t
   * @param stream see cuda::stream
   */
  void sort(sort::order_t order = sort::order_t::ascending,
            cuda::stream_t stream = 0) {
    // Bool-map frontier is always sorted.
  }

  void print() {
    std::cout << "Frontier = ";
    thrust::copy(storage.begin(),
                 storage.begin() + this->get_number_of_elements(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

 private:
  vector_t<int, memory_space_t::device> storage;
  std::size_t num_elements;  // number of elements in the frontier.
};

}  // namespace frontier
}  // namespace gunrock