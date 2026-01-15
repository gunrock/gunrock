#ifndef GUNROCK_COMPAT_CUDA_RUNTIME_API_H
#define GUNROCK_COMPAT_CUDA_RUNTIME_API_H

// Compatibility header for CUDA runtime API
// When building with NVCC (pure CUDA), use CUDA headers directly
// When building with HIP compiler, map CUDA to HIP equivalents

#if defined(__CUDACC__) && !defined(__HIP__)
  // Pure CUDA (NVCC without HIP): Use CUDA headers directly
  #include <cuda_runtime.h>
  // No mappings needed - CUDA APIs are used directly
#else
  // HIP compiler (works for both NVIDIA and AMD platforms)
  // Map CUDA types to HIP equivalents
  #include <hip/hip_runtime_api.h>
  
  #define cudaError_t hipError_t
  #define cudaSuccess hipSuccess
  #define cudaGetDevice hipGetDevice
  #define cudaSetDevice hipSetDevice
  #define cudaDeviceSynchronize hipDeviceSynchronize
  #define cudaGetLastError hipGetLastError
  #define cudaMemcpy hipMemcpy
  #define cudaMemcpyKind hipMemcpyKind
  #define cudaMemcpyHostToDevice hipMemcpyHostToDevice
  #define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
  #define cudaMalloc hipMalloc
  #define cudaFree hipFree
  #define cudaMemset hipMemset
#endif

#endif // GUNROCK_COMPAT_CUDA_RUNTIME_API_H
