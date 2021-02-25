#pragma once

#include <gunrock/util/error_utils.cuh>
#include <iostream>
#include <vector>
#include <unordered_map>

namespace gunrock {
namespace util {


/*
 * Use to Save the device state at the start of a function.
 * Restore() before returning.
 * Ideally, this would work with just the constructor and
 * the destructor would restore state automatically -- just
 * worried that an unused variable would get optimized away (need
 * to verify the behavior).
 */
struct SaveToRestore {
    int current_device;

    void Save() {
        // save the current device
        cudaGetDevice(&current_device);
    }

    void Restore() {
        // restore the current device
        cudaSetDevice(current_device);
    }

    SaveToRestore() = default;
    ~SaveToRestore() = default;
    SaveToRestore(const SaveToRestore& rhs) = default;
    SaveToRestore& operator=(const SaveToRestore& rhs) = default;
};

struct SingleGpuContext {

    int device_id;
    cudaStream_t stream;
    cudaEvent_t event;
    cudaDeviceProp prop;
    SaveToRestore state;

    // sdp: cudaFlags -- not sure I want to go this route, but it's a start
    SingleGpuContext(int deviceId, unsigned int cudaFlags = cudaEventDisableTiming) :
        device_id(deviceId) {

        state.Save();

        cudaSetDevice(device_id);
        cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
        cudaEventCreate(&event, cudaFlags);
        cudaGetDeviceProperties(&prop, device_id);

        state.Restore();
    }

    // sdp: not ideal, I'd rather have a destructor (and RAII), but
    // not sure how to avoid "accidental" destructor calls sans pointers.
    // Get rid of this "cudaError_t" asap
    cudaError_t Release() {
        cudaError_t retval = cudaSuccess;

        state.Save();

        GUARD_CU(cudaSetDevice(device_id));
        GUARD_CU(cudaStreamDestroy(stream));
        GUARD_CU(cudaEventDestroy(event));

        state.Restore();

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
    SaveToRestore state;

    // Map device to peers access enabled
    // (e.g., current_device -> p1,p2 --- p2 -> current_device, p1)
    // Makes disabling peer access easier because cuda doesn't report back if
    // devices are peers, but wll throw an error if they're not and you try to
    // cudaDeviceDisablePeerAccess.
    using PeerAccessMap = std::unordered_map<int, int>;
    PeerAccessMap device_peers;

    // Simply construct one context for each device
    // available on the system until we realize we need more flexibility.
    MultiGpuContext() {
        int device_count = 1;
        cudaGetDeviceCount(&device_count);
        contexts.reserve(device_count);
        for (int i = 0; i < device_count; i++) {
            contexts.push_back( SingleGpuContext(i) );
        }

        enablePeerAccess();
    }

    // sdp: not ideal, I'd rather have a destructor (and RAII), but
    // not sure how to avoid "accidental" destructor calls sans pointers.
    // Get rid of this "cudaError_t" asap
    cudaError_t Release() {
        cudaError_t retval = cudaSuccess;

        disablePeerAccess();

        for (auto& context : contexts ) {
            // could miss an error as written, but fine for now
            retval = context.Release();
        }

        contexts.clear();

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

    // Enable all-to-all peer access
    cudaError_t enablePeerAccess() {
        state.Save();
        cudaError_t retval = cudaSuccess;

        for (auto& context : contexts ) {
            for (auto& peer_context : contexts) {
                if (peer_context.device_id == context.device_id) {
                    continue;
                }

                int can_access_peer;
                GUARD_CU(cudaDeviceCanAccessPeer(&can_access_peer,
                                                 context.device_id,
                                                 peer_context.device_id));
                if (can_access_peer) {
                    GUARD_CU(cudaSetDevice(context.device_id));
                    GUARD_CU(cudaDeviceEnablePeerAccess(peer_context.device_id, 0));
                    device_peers.insert(std::make_pair(context.device_id, peer_context.device_id));
                }
                else {
                    std::cout << "WARNING! No peer access from "
                              << context.prop.name
                              << " (GPU" << context.device_id << ") -> "
                              << peer_context.prop.name
                              << " (GPU" << peer_context.device_id << ")\n";
                }
            }
        }

        state.Restore();

        return retval;
    }

    // Disable (assumed) all-to-all peer access
    cudaError_t disablePeerAccess() {
        state.Save();
        cudaError_t retval = cudaSuccess;

        for (const auto& kv: device_peers) {
            GUARD_CU(cudaSetDevice(kv.first));
            GUARD_CU(cudaDeviceDisablePeerAccess(kv.second));
        }

        // delete the map entries
        device_peers.clear();

        state.Restore();

        return retval;
    }
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
 *
 * Indicate function as inline to tell compiler/linker it can be included in
 * multiple files.
 */
inline int ceil_divide(int numerator, int divisor) {
    return (numerator + divisor - 1) / divisor;
}

} //namespace util
} //namespace gunrock
