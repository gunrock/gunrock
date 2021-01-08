/**
 * @file array.hxx
 * @author your name (you@domain.com)
 * @brief Dense data structure supported within gunrock. Note that the
 * array structure is modeled exactly after std::array, this is to make it
 * easier for users to familiarize themselves with the code.
 * gunrock::container::dense have further support for extended features that
 * are not available within the standard.
 *
 * Containers supported within gunrock, includes host and device support
 * for various useful data structures such as a basic static dense array
 * (std::array), sparse array and dynamic array. Ccontainers are the other
 * fundamental part of algorithms ( @see algo for the other half of what is
 * supported).
 *
 *
 * @todo extended array support for display, print (std::out and file::out),
 * etc. features. These have proven very useful in the gunrock v1 and prior
 * versions and we would like to continue to support them in the future.
 *
 * need to support iterators for dense::array.
 *
 * @version 0.1
 * @date 2020-10-05
 *
 * @copyright Copyright (c) 2020
 *
 */

#pragma once

#include <cstddef>
#include <cstdlib>
#include <array>
#include <tuple>
#include <utility>
#include <algorithm>
#include <iterator>
#include <type_traits>

namespace gunrock {

/**
 * @brief Aggregate type for array structure.
 *
 * @tparam T
 * @tparam NumElements
 */
template <typename T, std::size_t NumElements>
struct array_traits {
  typedef T type[NumElements];

  __host__ __device__ __forceinline__ static constexpr T& reference(
      const type& t,
      std::size_t n) noexcept {
    return const_cast<T&>(t[n]);
  }

  __host__ __device__ __forceinline__ static constexpr T* pointer(
      const type& t) noexcept {
    return const_cast<T*>(t);
  }
};

/**
 * @brief Used as null_type for array container.
 *
 * @tparam T
 */
template <typename T>
struct array_traits<T, 0> {
  struct type {};

  __host__ __device__ __forceinline__ static constexpr T& reference(
      const type&,
      std::size_t) noexcept {
    return *static_cast<T*>(nullptr);
  }

  __host__ __device__ __forceinline__ static constexpr T* pointer(
      const type&) noexcept {
    return nullptr;
  }
};

/**
 * @brief std::array<> based array container implementation with full
 * device-side support.
 *
 * @tparam T            Type of array elements.
 * @tparam NumElements  Number of elements in array.
 */
template <typename T, std::size_t NumElements>
struct array {
  using value_type = T;
  using pointer_t = value_type*;
  using const_pointer_t = const value_type*;
  using reference_t = value_type&;
  using const_reference_t = const value_type&;

  using iterator = value_type*;
  using const_iterator = const value_type*;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  using array_type = array_traits<value_type, NumElements>;

  typename array_type::type elements;

 public:
  /**
   * @brief Return pointer of array on host or device-side
   * @return pointer_t
   */
  __host__ __device__ __forceinline__ constexpr pointer_t data() noexcept {
    return array_type::pointer(elements);
  }

  /**
   * @brief Return a const pointer of array on host or device-side
   * @return const_pointer_t
   */
  __host__ __device__ __forceinline__ constexpr const_pointer_t data() const
      noexcept {
    return array_type::pointer(elements);
  }

  /**
   * @brief
   * @return size_type
   */
  __host__ __device__ __forceinline__ constexpr size_type size() const
      noexcept {
    return NumElements;
  }

  __host__ __device__ __forceinline__ constexpr size_type max_size() const
      noexcept {
    return NumElements;
  }

  __host__ __device__ __forceinline__ constexpr bool empty() const noexcept {
    return size() == 0;
  }

  /**
   * @brief
   *
   * @param n
   * @return reference_t
   */
  __host__ __device__ __forceinline__ constexpr reference_t operator[](
      size_type n) noexcept {
    return array_type::reference(elements, n);
  }

  __host__ __device__ __forceinline__ constexpr const_reference_t operator[](
      size_type n) const noexcept {
    return array_type::reference(elements, n);
  }

  /*
   * algorithms::
   * with this implementation, we can use thrust's support for
   * all algorithms. neat!
   */

};  // struct: array

}  // namespace gunrock