#ifndef GUNROCK_COMPAT_CUDA_RUNTIME_API_H
#define GUNROCK_COMPAT_CUDA_RUNTIME_API_H

// Compatibility header for CUDA runtime API in HIP
#include <hip/hip_runtime_api.h>

// Map CUDA types to HIP equivalents
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

#endif // GUNROCK_COMPAT_CUDA_RUNTIME_API_H
