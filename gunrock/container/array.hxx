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

#include <array>  // host-side array implementation

// includes: thrust
#include <thrust/device_vector.h>  // XXX: Replace with gunrock::vector<>

namespace gunrock {

template <typename T, std::size_t NumElements>
struct array {
  typedef T value_type;
  typedef value_type* pointer_t;
  typedef const value_type* const_pointer_t;
  typedef value_type& reference_t;
  typedef const value_type& const_reference_t;

  typedef value_type* iterator;
  typedef const value_type* const_iterator;
  typedef std::size_t size_type;
  typedef std::ptrdiff_t difference_type;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  typedef array<value_type, NumElements> array_t;

 private:
  thrust::device_vector<value_type> _storage;
  pointer_t _ptr;

 public:
  /*
   * constructor::
   */
  array() : _storage(NumElements), _ptr(_storage.data().get()) {}

  //  Define an empty destructor to explicitly specify
  //  its execution space qualifier.
  // __host__ ~array(void) {}

  /*
   * pointers::
   */

  // Return pointer of array on host or device-side
  __host__ __device__ __forceinline__ pointer_t data() noexcept { return _ptr; }

  // Return a const pointer of array on host or device-side
  __host__ __device__ __forceinline__ constexpr const_pointer_t data() const
      noexcept {
    return const_cast<const_pointer_t>(_ptr);
  }

  /*
   * capacity::
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

  /*
   * access::
   */

  // Since C++14, this may alsy be constexpr
  __host__ __device__ __forceinline__ reference_t
  operator[](size_type n) noexcept {
    return _ptr[n];
  }

  __host__ __device__ __forceinline__ value_type constexpr operator[](
      size_type n) const noexcept {
    return _ptr[n];
  }

  /*
   * algorithms::
   * with this implementation, we can use thrust's support for
   * all algorithms. neat!
   */

};  // struct: array

}  // namespace gunrock