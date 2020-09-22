#pragma once

#include <iostream>
#include <gunrock/error.hxx>

namespace gunrock {
namespace memory {

/**
 * @brief memory space; cuda (device) or host.
 * Can be extended to support uvm and multi-gpu.
 * 
 */
enum memory_space_t {
    device,
    host
};

/**
 * @brief allocate a pointer with size on a specfied memory space.
 * 
 * @tparam T 
 * @param size 
 * @param space 
 * @return T* 
 */
template<typename T>
T* allocate(std::size_t size, memory_space_t space) {
    void *p = nullptr;
    if(size) {
        error::error_t status = (device == space) ?
            cudaMalloc(&p, size) : cudaMallocHost(&p, size);
        if (status != cudaSuccess) throw error::exception_t(status);
    }

    return reinterpret_cast<T*>(p);
}

/**
 * @brief free allocated memory on memory space.
 * 
 * @tparam T 
 * @param p 
 * @param space 
 */
template<typename T>
void free(T* p, memory_space_t space) {
    if(p) {
        error::error_t status = (device == space) ? 
        cudaFree((void*)p) : cudaFreeHost((void*)p);    // XXX: reinterpret_cast?
        if(cudaSuccess != status) throw error::exception_t(status);
    }
}

} // namespace: memory
} // namespace: gunrock