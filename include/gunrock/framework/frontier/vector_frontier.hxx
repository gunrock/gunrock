/**
 * @file vector_frontier.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Vector-based frontier implementation.
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
#include <gunrock/util/load_store.hxx>

#include <thrust/sequence.h>

namespace gunrock {
namespace frontier {
using namespace memory;

template <typename type_t>
class vector_frontier_t {
 public:
  using pointer_t = type_t*;

  // Constructors
  vector_frontier_t() : storage(), ptr(storage.data().get()), num_elements(0) {
    std::cout << "Empty C -- Pointer: " << ptr << std::endl;
  }
  vector_frontier_t(std::size_t size)
      : storage(size),
        // ptr(storage.data().get()),
        num_elements(size) {
    // std::cout << "Pointer: " << ptr << std::endl;
  }

  // Empty Destructor, this is important on kernel-exit.
  ~vector_frontier_t() {}

  // Copy Constructor
  vector_frontier_t(const vector_frontier_t& other)
      : storage(other.storage),
        ptr(other.ptr),
        num_elements(other.num_elements) {
    printf("Copy constructor called at vector_frontier_t\n");
  }

  // Disable move and assignment.
  vector_frontier_t& operator=(const vector_frontier_t& rhs) = delete;
  vector_frontier_t& operator=(vector_frontier_t&&) = delete;
  vector_frontier_t(vector_frontier_t&&) = delete;

  /**
   * @brief Get the number of elements within the frontier.
   * @return std::size_t
   */
  std::size_t get_number_of_elements(cuda::stream_t stream = 0) const {
    return num_elements;
  }

  /**
   * @brief Get the capacity (number of elements possible).
   * @return std::size_t
   */
  std::size_t get_capacity() const { return storage.capacity(); }

  /**
   * @brief Get the element at the specified index.
   *
   * @param idx
   * @return type_t
   */
  __device__ __forceinline__ type_t
  get_element_at(std::size_t const& idx) const {
    auto element = thread::load(ptr + idx);
    return element;
  }

  /**
   * @brief Set the element at the specified index.
   *
   * @param idx
   * @param element
   * @return void
   */
  __device__ __forceinline__ void set_element_at(std::size_t const& idx,
                                                 type_t const& element) {
    thread::store(ptr + idx, element);
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

  pointer_t data() { return raw_pointer_cast(storage.data()) /* .get() */; }
  __host__ __device__ __forceinline__ pointer_t get() const { return ptr; }
  pointer_t begin() { return this->data(); }
  pointer_t end() { return this->begin() + this->get_number_of_elements(); }
  bool is_empty() const { return (this->get_number_of_elements() == 0); }

  /**
   * @brief (vertex-like) push back a value to the frontier.
   *
   * @param value
   */
  void push_back(type_t const& value) {
    storage.push_back(value);
    num_elements++;
  }

  /**
   * @brief Fill the entire frontier with a user-specified value.
   *
   * @param value
   * @param stream
   */
  void fill(type_t const value, cuda::stream_t stream = 0) {
    thrust::fill(thrust::cuda::par.on(stream), this->begin(), this->end(),
                 value);
  }

  /**
   * @brief `sequence` fills the entire frontier with a sequence of numbers.
   *
   * @param initial_value The first value of the sequence.
   * @param size Number of elements to fill the sequence up to. Also corresponds
   * to the new size of the frontier.
   * @param stream @see `cuda::stream_t`.
   *
   * @todo Maybe we should accept `standard_context_t` instead of `stream_t`.
   */
  void sequence(type_t const initial_value,
                std::size_t const& size,
                cuda::stream_t stream = 0) {
    // Resize if needed.
    if (this->get_capacity() < size)
      this->reserve(size);

    // Set the new number of elements.
    this->set_number_of_elements(size);

    thrust::sequence(thrust::cuda::par.on(stream), this->begin(), this->end(),
                     initial_value);
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
  void resize(
      std::size_t const& size,
      type_t const default_value = gunrock::numeric_limits<type_t>::invalid()) {
    storage.resize(size, default_value);
    if (ptr != nullptr)
      ptr = raw_pointer_cast(storage.data());
    std::cout << "resize Pointer: " << ptr << std::endl;
  }

  /**
   * @brief "Hints" the alocator that we need to reserve the suggested size. The
   * capacity() will increase and report reserved() size, but size() will still
   * report the actual size, not reserved size. See std::vector for more detail.
   *
   * @param size size to reserve (size is in count not bytes).
   */
  void reserve(std::size_t const& size) {
    storage.reserve(size);
    ptr = raw_pointer_cast(storage.data());
    std::cout << "reserve Pointer: " << ptr << std::endl;
  }

  /**
   * @brief Parallel sort the frontier.
   *
   * @param order see sort::order_t
   * @param stream see cuda::stream
   */
  void sort(sort::order_t order = sort::order_t::ascending,
            cuda::stream_t stream = 0) {
    sort::radix::sort_keys(storage.data().get(), this->get_number_of_elements(),
                           order, stream);
  }

  void print() {
    std::cout << "Frontier = ";
    thrust::copy(storage.begin(),
                 storage.begin() + this->get_number_of_elements(),
                 std::ostream_iterator<type_t>(std::cout, " "));
    std::cout << std::endl;
  }

 private:
  vector_t<type_t, memory_space_t::device> storage;
  pointer_t ptr;
  std::size_t num_elements;  // number of elements in the frontier.
};

}  // namespace frontier
}  // namespace gunrock