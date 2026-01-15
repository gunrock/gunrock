/**
 * @file runtime_api.h
 * @brief Compatibility layer for HIP/CUDA runtime APIs
 * 
 * This header provides a unified interface for both HIP and CUDA runtime APIs.
 * When building with NVIDIA backend, HIP APIs are mapped to CUDA equivalents.
 * When building with AMD backend, HIP APIs are used directly.
 */

#ifndef GUNROCK_COMPAT_RUNTIME_API_H
#define GUNROCK_COMPAT_RUNTIME_API_H

// When using NVCC directly (pure CUDA), map HIP APIs to CUDA
// When using HIP compiler (with NVIDIA or AMD platform), use HIP APIs
#if defined(__CUDACC__) && !defined(__HIP__)
  // Pure CUDA (NVCC without HIP): Map HIP APIs to CUDA
  #include <cuda_runtime.h>
  
  // Map HIP types to CUDA types
  typedef cudaStream_t hipStream_t;
  typedef cudaEvent_t hipEvent_t;
  typedef cudaError_t hipError_t;
  typedef cudaDeviceProp hipDeviceProp_t;  // Note: CUDA uses struct cudaDeviceProp, not cudaDeviceProp_t
  typedef cudaMemcpyKind hipMemcpyKind;
  
  // hipDeviceptr_t will be defined later for CUDA 10.2+ (virtual memory APIs)
  // For older versions or if CUDA version check fails, use void*
  #if !(defined(CUDART_VERSION) && CUDART_VERSION >= 10020)
    typedef void* hipDeviceptr_t;
  #endif
  
  // Map HIP constants to CUDA constants
  #define hipSuccess cudaSuccess
  #define hipErrorUnknown cudaErrorUnknown
  #define hipStreamNonBlocking cudaStreamNonBlocking
  #define hipEventDisableTiming cudaEventDisableTiming
  #define hipMemcpyHostToDevice cudaMemcpyHostToDevice
  #define hipMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define hipMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
  
  // Map HIP functions to CUDA functions
  #define hipSetDevice cudaSetDevice
  #define hipGetDevice cudaGetDevice
  #define hipGetDeviceCount cudaGetDeviceCount
  #define hipGetDeviceProperties cudaGetDeviceProperties
  #define hipDeviceSynchronize cudaDeviceSynchronize
  #define hipStreamSynchronize cudaStreamSynchronize
  #define hipStreamCreateWithFlags cudaStreamCreateWithFlags
  #define hipStreamDestroy cudaStreamDestroy
  #define hipEventCreate cudaEventCreate
  #define hipEventCreateWithFlags cudaEventCreateWithFlags
  #define hipEventDestroy cudaEventDestroy
  #define hipEventRecord cudaEventRecord
  #define hipEventSynchronize cudaEventSynchronize
  #define hipEventElapsedTime cudaEventElapsedTime
  #define hipGetLastError cudaGetLastError
  inline const char* hipGetErrorString(cudaError_t error) { return cudaGetErrorString(error); }
  #define hipMemcpy cudaMemcpy
  #define hipMalloc cudaMalloc
  #define hipFree cudaFree
  #define hipHostMalloc cudaMallocHost
  #define hipHostFree cudaFreeHost
  #define hipMemset cudaMemset
  #define hipMemGetInfo cudaMemGetInfo
  #define hipDriverGetVersion cudaDriverGetVersion
  #define hipRuntimeGetVersion cudaRuntimeGetVersion
  #define hipDeviceEnablePeerAccess cudaDeviceEnablePeerAccess
  #define hipDeviceGetAttribute cudaDeviceGetAttribute
  #define hipDeviceAttributeMaxGridDimX cudaDevAttrMaxGridDimX
  #define hipDeviceAttributeClockRate cudaDevAttrClockRate
  #define hipDeviceAttributeMemoryClockRate cudaDevAttrMemoryClockRate
  
  // Function attributes
  typedef cudaFuncAttributes hipFuncAttributes;
  #define hipFuncGetAttributes cudaFuncGetAttributes
  
  // Function cache preferences
  typedef cudaFuncCache hipFuncCache_t;
  #define hipFuncCachePreferNone cudaFuncCachePreferNone
  #define hipFuncCachePreferShared cudaFuncCachePreferShared
  #define hipFuncCachePreferL1 cudaFuncCachePreferL1
  #define hipFuncCachePreferEqual cudaFuncCachePreferEqual
  
  // Shared memory config
  typedef cudaSharedMemConfig hipSharedMemConfig;
  #define hipSharedMemBankSizeDefault cudaSharedMemBankSizeDefault
  #define hipSharedMemBankSizeFourByte cudaSharedMemBankSizeFourByte
  #define hipSharedMemBankSizeEightByte cudaSharedMemBankSizeEightByte
  
  // Occupancy calculation
  #define hipOccupancyMaxActiveBlocksPerMultiprocessor cudaOccupancyMaxActiveBlocksPerMultiprocessor
  
  // Cooperative kernel launch (CUDA 9.0+)
  #define hipLaunchCooperativeKernel cudaLaunchCooperativeKernel
  #define hipLaunchCooperativeKernelMultiDevice cudaLaunchCooperativeKernelMultiDevice
  
  // Virtual memory APIs (CUDA 10.2+, may not be available in older versions)
  #if defined(CUDART_VERSION) && CUDART_VERSION >= 10020
    #include <cuda.h>
    
    typedef CUmemGenericAllocationHandle hipMemGenericAllocationHandle_t;
    typedef CUmemAllocationProp hipMemAllocationProp;
    typedef CUmemAccessDesc hipMemAccessDesc;
    typedef CUmemLocation hipMemLocation;
    typedef CUmemAllocationGranularity_flags hipMemAllocationGranularity_flags;
    typedef CUmemAccess_flags hipMemAccess_flags;
    typedef CUmemLocationType hipMemLocationType;
    
    // Define hipDeviceptr_t as CUdeviceptr for CUDA 10.2+ virtual memory APIs
    typedef CUdeviceptr hipDeviceptr_t;
    
    #define hipMemGetAllocationGranularity cuMemGetAllocationGranularity
    #define hipMemCreate cuMemCreate
    #define hipMemRelease cuMemRelease
    #define hipMemAddressReserve cuMemAddressReserve
    #define hipMemAddressFree cuMemAddressFree
    #define hipMemMap cuMemMap
    #define hipMemSetAccess cuMemSetAccess
    #define hipMemUnmap cuMemUnmap
    
    #define hipMemAllocationGranularityMinimum CU_MEM_ALLOCATION_GRANULARITY_MINIMUM
    #define hipMemLocationTypeDevice CU_MEM_LOCATION_TYPE_DEVICE
    #define hipMemAccessFlagsProtNone CU_MEM_ACCESS_FLAGS_PROT_NONE
    #define hipMemAccessFlagsProtRead CU_MEM_ACCESS_FLAGS_PROT_READ
    #define hipMemAccessFlagsProtReadWrite CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    
    // Note: These struct definitions may need adjustment based on CUDA version
    // The HIP structs should match CUDA structs exactly
  #else
    // Virtual memory APIs not available in older CUDA versions
    // Define minimal stubs or mark as unavailable
    typedef void* hipDeviceptr_t;  // Define it here if not already defined
    typedef void* hipMemGenericAllocationHandle_t;
    typedef void* hipMemAllocationProp;
    typedef void* hipMemAccessDesc;
    typedef void* hipMemLocation;
    typedef int hipMemAllocationGranularity_flags;
    typedef int hipMemAccess_flags;
    typedef int hipMemLocationType;
    
    #define hipMemGetAllocationGranularity(...) (cudaErrorNotSupported)
    #define hipMemCreate(...) (cudaErrorNotSupported)
    #define hipMemRelease(...) (cudaErrorNotSupported)
    #define hipMemAddressReserve(...) (cudaErrorNotSupported)
    #define hipMemAddressFree(...) (cudaErrorNotSupported)
    #define hipMemMap(...) (cudaErrorNotSupported)
    #define hipMemSetAccess(...) (cudaErrorNotSupported)
    #define hipMemUnmap(...) (cudaErrorNotSupported)
    
    #define hipMemAllocationGranularityMinimum 0
    #define hipMemLocationTypeDevice 0
    #define hipMemAccessFlagsProtNone 0
    #define hipMemAccessFlagsProtRead 1
    #define hipMemAccessFlagsProtReadWrite 3
  #endif
#else
  // HIP compiler (works for both NVIDIA and AMD platforms)
  #include <hip/hip_runtime.h>
  #include <hip/hip_runtime_api.h>
#endif

#endif // GUNROCK_COMPAT_RUNTIME_API_H
