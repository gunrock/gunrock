#pragma once

#include <iostream>
#include <memory>

#include <thrust/device_ptr.h>
#include <gunrock/error.hxx>

namespace gunrock {
namespace memory {

/**
 * @brief memory space; cuda (device) or host.
 * Can be extended to support uvm and multi-gpu.
 *
 * @todo change this enum to support cudaMemoryType
 * (see ref;  std::underlying_type<cudaMemoryType>::type)
 * instead of some random enums, we can rely
 * on cudaMemoryTypeHost/Device/Unregistered/Managed
 * for this.
 *
 */
enum memory_space_t { device, host };

/**
 * @brief allocate a pointer with size on a specfied memory space.
 *
 * @tparam T return type of the pointer being allocated.
 * @param size size in bytes (bytes to be allocated).
 * @param space memory space domain where the allocation should happen.
 * @return T* return allocated T* pointer.
 */
template <typename type_t>
inline type_t* allocate(std::size_t size, memory_space_t space) {
  void* pointer = nullptr;
  if (size) {
    error::error_t status = (device == space) ? cudaMalloc(&pointer, size)
                                              : cudaMallocHost(&pointer, size);
    error::throw_if_exception(status);
  }

  return reinterpret_cast<type_t*>(pointer);
}

/**
 * @brief free allocated memory on memory space.
 *
 * @tparam T type of the pointer.
 * @param pointer pointer to free memory of.
 * @param space memory space domain where the pointer was allocated.
 */
template <typename type_t>
inline void free(type_t* pointer, memory_space_t space) {
  if (pointer) {
    error::error_t status = (device == space) ? cudaFree((void*)pointer)
                                              : cudaFreeHost((void*)pointer);
    error::throw_if_exception(status);
  }
}

/**
 * @brief Wrapper around thrust::raw_pointer_cast() to accept .data() or raw
 * pointer and return a raw pointer. Useful when we would like to return a raw
 * pointer of either a thrust device vector or a host vector. Because thrust
 * device vector's raw pointer is accessed by `.data().get()`, whereas thrust
 * host vector's raw pointer is simply `data()`. So, when calling these
 * functions on `.data()`, it can cast either a host or device vector.
 *
 * @tparam type_t
 * @param pointer
 * @return type_t*
 */
template <typename type_t>
inline type_t* raw_pointer_cast(thrust::device_ptr<type_t> pointer) {
  return thrust::raw_pointer_cast(pointer);
}

/**
 * @brief Wrapper around thrust::raw_pointer_cast() to accept .data() or raw
 * pointer and return a raw pointer. Useful when we would like to return a raw
 * pointer of either a thrust device vector or a host vector. Because thrust
 * device vector's raw pointer is accessed by `.data().get()`, whereas thrust
 * host vector's raw pointer is simply `data()`. So, when calling these
 * functions on `.data()`, it can cast either a host or device vector.
 *
 * @tparam type_t
 * @param pointer
 * @return type_t*
 */
template <typename type_t>
__host__ __device__ inline type_t* raw_pointer_cast(type_t* pointer) {
  return thrust::raw_pointer_cast(pointer);
}

}  // namespace memory
}  // namespace gunrock