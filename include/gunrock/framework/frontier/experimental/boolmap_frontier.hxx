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
namespace experimental {
namespace frontier {
using namespace memory;

template <typename type_t>
class boolmap_frontier_t {
 public:
  using pointer_t = type_t*;

  // Constructors
  boolmap_frontier_t() : num_elements(0) {
    p_storage = std::make_shared<vector_t<type_t, memory_space_t::device>>(
        vector_t<type_t, memory_space_t::device>(1));
  }
  boolmap_frontier_t(std::size_t size) : num_elements(size) {
    p_storage = std::make_shared<vector_t<type_t, memory_space_t::device>>(
        vector_t<type_t, memory_space_t::device>(size));
    raw_ptr = nullptr;
  }

  // Empty Destructor, this is important on kernel-exit.
  ~boolmap_frontier_t() {}

  // Copy Constructor
  boolmap_frontier_t(const boolmap_frontier_t& rhs) {
    p_storage = rhs.p_storage;
    raw_ptr = rhs.p_storage.get()->data().get();
    num_elements = rhs.num_elements;
  }

  /**
   * @brief Get the number of elements within the frontier. This is a costly
   * feature of a boolmap and should be avoided when possible.
   * @return std::size_t
   */
  __host__ __device__ __forceinline__ std::size_t get_number_of_elements(
      gcuda::stream_t stream = 0) {
    // Compute number of elements using a reduction.
#ifdef __CUDA_ARCH__
    num_elements = thrust::reduce(thrust::seq, this->begin(), this->end(), 0);
#else
    num_elements = thrust::reduce(thrust::cuda::par.on(stream), this->begin(),
                                  this->end(), 0);
#endif

    return num_elements;
  }

  /**
   * @brief Get the capacity (number of elements possible).
   * @return std::size_t
   */
  std::size_t get_capacity() const { return p_storage.get()->capacity(); }

  /**
   * @brief Get the element at the specified index.
   *
   * @param idx
   * @return type_t
   */
  __device__ __forceinline__ constexpr const type_t get_element_at(
      std::size_t const& idx) const noexcept {
    return this->get()[idx] == 1 ? idx
                                 : gunrock::numeric_limits<type_t>::invalid();
  }

  __device__ __forceinline__ constexpr type_t get_element_at(
      std::size_t const& idx) noexcept {
    return this->get()[idx] == 1 ? idx
                                 : gunrock::numeric_limits<type_t>::invalid();
  }

  /**
   * @brief Set the element at the specified index.
   *
   * @param idx
   * @param element
   * @return void
   */
  __device__ __forceinline__ constexpr void set_element_at(
      type_t const& element,
      std::size_t const& idx = 0  // Ignore idx for boolmap.
  ) const noexcept {              // XXX: This should not be const
    thread::store(this->get() + element, 1);
  }

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

  /**
   * @brief Access to internal raw pointer, works on host and device.
   */
  __host__ __device__ __forceinline__ constexpr pointer_t get() const {
    return raw_ptr;
  }

  pointer_t data() { return raw_pointer_cast(p_storage.get()->data()); }
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
  void fill(type_t const value, gcuda::stream_t stream = 0) {
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
    p_storage.get()->resize(size, default_value);
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
  void reserve(std::size_t const& size) { p_storage.get()->reserve(size); }

  /**
   * @brief Parallel sort the frontier.
   *
   * @param order see sort::order_t
   * @param stream see gcuda::stream
   */
  void sort(sort::order_t order = sort::order_t::ascending,
            gcuda::stream_t stream = 0) {
    // Bool-map frontier is always sorted.
  }

  void print() {
    std::cout << "Frontier = ";
    thrust::copy(p_storage.get()->begin(),
                 p_storage.get()->begin() + this->get_number_of_elements(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

 private:
  std::shared_ptr<vector_t<type_t, memory_space_t::device>> p_storage;
  pointer_t raw_ptr;
  std::size_t num_elements;  // number of elements in the frontier.
};

}  // namespace frontier
}  // namespace experimental
}  // namespace gunrock