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

#include <gunrock/framework/frontier/configs.hxx>

#include <gunrock/util/type_limits.hxx>
#include <gunrock/util/load_store.hxx>

#include <gunrock/container/vector.hxx>
#include <gunrock/algorithms/sort/radix_sort.hxx>

#include <thrust/sequence.h>

namespace gunrock {
namespace frontier {
using namespace memory;

template <typename vertex_t, typename edge_t, frontier_kind_t _kind>
class vector_frontier_t {
 public:
  using vector_frontier_type = vector_frontier_t<vertex_t, edge_t, _kind>;
  using type_t = std::conditional_t<_kind == frontier_kind_t::vertex_frontier,
                                    vertex_t,
                                    edge_t>;

  /**
   * @brief Default constructor.
   */
  vector_frontier_t()
      : num_elements(0), raw_ptr(nullptr), resizing_factor(1.0f) {
    /// TODO: we are using a vector of size 1 to avoid the overhead of setting
    /// it up later. Check if this is valid to do.
    p_storage = std::make_shared<vector_t<type_t, memory_space_t::device>>(
        vector_t<type_t, memory_space_t::device>());
  }

  /**
   * @brief Construct a new vector frontier object with a given size and
   * frontier resizing factor.
   *
   * @param size
   * @param frontier_resizing_factor
   */
  vector_frontier_t(std::size_t size, float frontier_resizing_factor = 1.0f)
      : num_elements(size), resizing_factor(frontier_resizing_factor) {
    p_storage = std::make_shared<vector_t<type_t, memory_space_t::device>>(
        vector_t<type_t, memory_space_t::device>(size));
    raw_ptr = p_storage.get()->data().get();
  }

  // Empty Destructor, this is important on kernel-exit.
  ~vector_frontier_t() {}

  // Copy Constructor
  __device__ __host__ vector_frontier_t(const vector_frontier_t& rhs) {
#ifdef __CUDA_ARCH__
    raw_ptr = rhs.raw_ptr;
#else
    p_storage = rhs.p_storage;
    raw_ptr = rhs.p_storage.get()->data().get();
#endif
    num_elements = rhs.num_elements;
    resizing_factor = rhs.resizing_factor;
  }

  /**
   * @brief Get the number of elements within the frontier.
   * @return std::size_t
   */
  __host__ __device__ __forceinline__ std::size_t get_number_of_elements(
      gcuda::stream_t stream = 0) const {
    return num_elements;
  }

  /**
   * @brief Get the capacity (number of elements possible).
   * @return std::size_t
   */
  std::size_t get_capacity() const { return p_storage.get()->capacity(); }

  /**
   * @brief Get the resizing factor used to scale the frontier size.
   *
   * @return float
   */
  float get_resizing_factor() const { return resizing_factor; }

  /**
   * @brief Get the element at the specified index.
   *
   * @param idx
   * @return type_t
   */
  __device__ __forceinline__ constexpr const type_t get_element_at(
      std::size_t const& idx) const noexcept {
    return thread::load(this->get() + idx);
  }

  __device__ __forceinline__ constexpr type_t get_element_at(
      std::size_t const& idx) noexcept {
    return thread::load(this->get() + idx);
  }

  /**
   * @brief Set the element at the specified index.
   *
   * @param idx
   * @param element
   */
  __device__ __forceinline__ constexpr void set_element_at(
      type_t const& element,
      std::size_t const& idx)
      const noexcept {  /// XXX: This should not be const
    thread::store(this->get() + idx, element);
  }

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
    num_elements = elements;
  }

  /**
   * @brief Access to internal raw pointer, works on host and device.
   */
  __host__ __device__ __forceinline__ constexpr auto get() const {
    return raw_ptr;
  }

  auto data() { return raw_pointer_cast(p_storage.get()->data()); }
  auto begin() { return this->data(); }
  auto end() { return this->begin() + this->get_number_of_elements(); }
  bool is_empty() const { return (this->get_number_of_elements() == 0); }

  /**
   * @brief (vertex-like) push back a value to the frontier.
   *
   * @param value
   */
  void push_back(type_t const& value) {
    p_storage.get()->push_back(value);
    num_elements++;
  }

  /**
   * @brief Fill the entire frontier with a user-specified value.
   *
   * @param value
   * @param stream
   */
  void fill(type_t const value, gcuda::stream_t stream = 0) {
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
                gcuda::stream_t stream = 0) {
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
    p_storage.get()->resize(size, default_value);
  }

  /**
   * @brief "Hints" the alocator that we need to reserve the suggested size. The
   * capacity() will increase and report reserved() size, but size() will still
   * report the actual size, not reserved size. See std::vector for more detail.
   *
   * @param size size to reserve (size is in count not bytes).
   */
  void reserve(std::size_t const& size) {
    p_storage.get()->reserve(size * resizing_factor);
  }

  /**
   * @brief Parallel sort the frontier.
   *
   * @param order see sort::order_t
   * @param stream see gcuda::stream
   */
  void sort(sort::order_t order = sort::order_t::ascending,
            gcuda::stream_t stream = 0) {
    sort::radix::sort_keys(p_storage.get()->data().get(),
                           this->get_number_of_elements(), order, stream);
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
  type_t* raw_ptr;
  std::size_t num_elements;  // number of elements in the frontier.
  float resizing_factor;     // reserve size * factor.
};

}  // namespace frontier
}  // namespace gunrock