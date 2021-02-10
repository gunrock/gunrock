#pragma once

#include <gunrock/util/error_utils.cuh>
#include <iostream>
#include <vector>

namespace gunrock {
namespace util {

struct SingleGpuContext {

    int device_id;
    cudaStream_t stream;
    cudaEvent_t event;

    // sdp: cudaFlags -- not sure I want to go this route, but it's a start
    SingleGpuContext(int deviceId, unsigned int cudaFlags = cudaEventDisableTiming) : 
        device_id(deviceId) {
        
        // save the current device
        int current_device;
        cudaGetDevice(&current_device);

        cudaSetDevice(device_id);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaEventCreate(&event, cudaFlags);

        // restore the current device
        cudaSetDevice(current_device);
    }

    // sdp: not ideal, I'd rather have a destructor (and RAII), but
    // not sure how to avoid "accidental" destructor calls sans pointers.
    // Get rid of this "cudaError_t" asap
    cudaError_t Release() {
        cudaError_t retval = cudaSuccess;

        // save the current device
        int current_device;
        GUARD_CU(cudaGetDevice(&current_device));

        GUARD_CU(cudaSetDevice(device_id));
        GUARD_CU(cudaStreamDestroy(stream));
        GUARD_CU(cudaEventDestroy(event));

        // restore the current device
        GUARD_CU(cudaSetDevice(current_device));

        return retval;
    }

    // default copy constructor / operator= / destrcutor for now
    SingleGpuContext(const SingleGpuContext& rhs) = default;
    SingleGpuContext& operator=(const SingleGpuContext& rhs) = default;
    ~SingleGpuContext() = default;

    // Output
    friend std::ostream& operator<<(std::ostream& os, const SingleGpuContext& context) {
        os << "device_id: " << context.device_id << "\n"
            << "stream: " << context.stream << "\n"
            << "event: " << context.event;
        
        return os;
    }
};



struct MultiGpuContext {
    std::vector<SingleGpuContext> contexts;

    // Simply construct one context for each device
    // available on the system until we realize we need more flexibility.
    MultiGpuContext() {
        int device_count = 1;
        cudaGetDeviceCount(&device_count);
        contexts.reserve(device_count);
        for (int i = 0; i < device_count; i++) {
            contexts.push_back( SingleGpuContext(i) );
        }
    }

    // sdp: not ideal, I'd rather have a destructor (and RAII), but
    // not sure how to avoid "accidental" destructor calls sans pointers.
    // Get rid of this "cudaError_t" asap
    cudaError_t Release() {
        cudaError_t retval = cudaSuccess;

        for (auto& context : contexts ) {
            // could miss an error as written, but fine for now
            retval = context.Release(); 
        }

        return retval;
    }

    // default copy constructor / operator= / destrcutor for now
    MultiGpuContext(const MultiGpuContext& rhs) = default;
    MultiGpuContext& operator=(const MultiGpuContext& rhs) = default;
    ~MultiGpuContext() = default;

    // Output
    friend std::ostream& operator<<(std::ostream& os, const MultiGpuContext& mgpu_context) {
        for (auto& context : mgpu_context.contexts) {
            os << "{\n" << context << "\n}\n";
        }
        return os;
    }

    int getGpuCount() const { return contexts.size(); }
};

// Data necessary to perform a multi-gpu forall
struct MultiGpuInfo {

    // sdp: might want to remove this, but maybe better to keep if
    // it makes sense to avoid sending along a gpu context too?
    cudaStream_t stream;
    cudaEvent_t event;

    // At the moment (Feb 2021), arrays split across GPUs are
    // 0-indexed on each GPU, but lambda functions need access to 
    // a global index that considers the array in aggregate. This 
    // offset provides that mapping / transformation 
    // (e.g., global_index = zero_based_index + offset).
    int offset;

    // number of elements
    int data_length;
};

/*
 * Standard integer division results in the floor of the operation.
 * This function gives us the cieling version of integer division.
 */
int ceil_divide(int numerator, int divisor) {
    return (numerator + divisor - 1) / divisor;
}

} //namespace util
} //namespace gunrock