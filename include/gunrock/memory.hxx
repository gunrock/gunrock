/**
 * @file memory.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2020-09-22
 *
 * @copyright Copyright (c) 2021
 *
 */

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
 * @brief allocate memory on defined memory space on a specific pointer.
 *
 * @tparam type_t type of the pointer
 * @param pointer pointer to the memory
 * @param size size of the memory in bytes
 * @param space memory space (device or host)
 */
template <typename type_t>
void allocate(type_t* pointer,
              std::size_t size,
              memory_space_t space = memory_space_t::device) {
  if (size) {
    error::throw_if_exception((device == space)
                                  ? cudaMalloc(&pointer, size)
                                  : cudaMallocHost(&pointer, size));
  }
}

/**
 * @brief allocate a pointer with size on a specfied memory space.
 *
 * @tparam T return type of the pointer being allocated.
 * @param size size in bytes (bytes to be allocated).
 * @param space memory space domain where the allocation should happen.
 * @return T* return allocated T* pointer.
 */
template <typename type_t>
inline type_t* allocate(std::size_t size,
                        memory_space_t space = memory_space_t::device) {
  void* pointer = nullptr;
  if (size) {
    error::throw_if_exception((device == space)
                                  ? cudaMalloc(&pointer, size)
                                  : cudaMallocHost(&pointer, size));
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
inline void free(type_t* pointer,
                 memory_space_t space = memory_space_t::device) {
  if (pointer) {
    error::throw_if_exception((device == space) ? cudaFree((void*)pointer)
                                                : cudaFreeHost((void*)pointer));
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

/**
 * @brief Custom deleter supports deletion of memory on device.
 *
 * @tparam type_t type of the pointer's memory to be deleted.
 */
template <typename type_t>
struct deleter_t {
  /**
   * @brief Free memory on device.
   *
   * @param pointer
   */
  void operator()(type_t* pointer) const { free(pointer); }
};

/**
 * @brief Custom allocator supports allocation of memory on device.
 *
 * @tparam type_t type of the pointer's memory to be allocated.
 */
template <typename type_t>
struct allocator_t {
  /**
   * @brief Allocate memory on device.
   *
   * @param size size in bytes.
   * @return type_t* returns the allocated pointer.
   */
  type_t* operator()(size_t bytes) const { allocate<type_t>(bytes); }
};

}  // namespace memory
}  // namespace gunrock