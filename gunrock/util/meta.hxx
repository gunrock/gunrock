/**
 * @see
 * https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#compilation-phases
 * Guiding CUDA architecture's compilation paths.
 *
 */
#ifdef __CUDA_ARCH__    // Device-side compilation
#define GUNROCK_CUDA_ARCH __CUDA_ARCH__
#endif

#define GUNROCK_CUDACC __CUDACC__   // Use nvcc to compile
#define GUNROCK_DEBUG __CUDACC_DEBUG__  // Library-wide debug flag

#ifdef GUNROCK_CUDACC

#ifndef GUNROCK_HOST_DEVICE
#define GUNROCK_HOST_DEVICE __forceinline__ __device__ __host__
#endif

#ifndef GUNROCK_DEVICE
#define GUNROCK_DEVICE __device__
#endif

// Requires --extended-lambda (-extended-lambda) flag.
// Allow __host__, __device__ annotations in lambda declarations.
#ifndef GUNROCK_LAMBDA
#define GUNROCK_LAMBDA __device__ __host__
#endif

#else  // #ifndef GUNROCK_CUDACC
#endif // #ifdef GUNROCK_CUDACC