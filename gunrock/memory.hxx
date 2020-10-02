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
enum memory_space_t {
    device,
    host,
    unknown
};

/**
 * @brief allocate a pointer with size on a specfied memory space.
 * 
 * @tparam T return type of the pointer being allocated.
 * @param size size in bytes (bytes to be allocated).
 * @param space memory space domain where the allocation should happen.
 * @return T* return allocated T* pointer. 
 */
template<typename type_t>
inline type_t* allocate(std::size_t size, memory_space_t space) {
    void *pointer = nullptr;
    if(space == unknown) throw error::exception_t(cudaErrorUnknown);
    if(size) {
        error::error_t status = (device == space) ?
            cudaMalloc(&pointer, size) : cudaMallocHost(&pointer, size);
        if (status != cudaSuccess) throw error::exception_t(status);
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
template<typename type_t>
inline void free(type_t* pointer, memory_space_t space) {
    if(space == unknown) throw error::exception_t(cudaErrorUnknown);
    if(pointer) {
        error::error_t status = (device == space) ? 
        cudaFree((void*)pointer) : cudaFreeHost((void*)pointer);    // XXX: reinterpret_cast?
        if(cudaSuccess != status) throw error::exception_t(status);
    }
}

template<typename type_t>
inline type_t* raw_pointer_cast(thrust::device_ptr<type_t> pointer) {
    return thrust::raw_pointer_cast(pointer);
}

template<typename type_t>
inline type_t* raw_pointer_cast(type_t* pointer) {
    return thrust::raw_pointer_cast(pointer);
}

} // namespace: memory
} // namespace: gunrock