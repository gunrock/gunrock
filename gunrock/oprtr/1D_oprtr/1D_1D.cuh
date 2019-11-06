// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * 1D_1D.cuh
 *
 * @brief 1D array with 1D array operations
 */

#pragma once

#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>

namespace gunrock {
namespace oprtr {

/**
 * @brief Add the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Set_Kernel(ValueT *d_dst, T *d_src, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_dst[idx] = d_src[idx];
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Set(ValueT *elements, T *values, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements, values,
      [] __host__ __device__(ValueT & element, T value) { element = value; },
      length, target, stream);
}

/**
 * @brief Add the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Add_Kernel(ValueT *d_dst, T *d_src, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_dst[idx] += d_src[idx];
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Add(ValueT *elements, T *values, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements, values,
      [] __host__ __device__(ValueT & element, T value) { element += value; },
      length, target, stream);
}

/**
 * @brief Minus the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Minus_Kernel(ValueT *d_dst, T *d_src, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_dst[idx] -= d_src[idx];
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Minus(ValueT *elements, T *values, SizeT length,
                  util::Location target = util::DEVICE,
                  cudaStream_t stream = 0) {
  return ForEach(
      elements, values,
      [] __host__ __device__(ValueT & element, T value) { element -= value; },
      length, target, stream);
}

/**
 * @brief Multiply the source vector to the destination vector with the same
 * length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Mul_Kernel(ValueT *d_dst, T *d_src, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_dst[idx] *= d_src[idx];
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Mul(ValueT *elements, T *values, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements, values,
      [] __host__ __device__(ValueT & element, T value) { element *= value; },
      length, target, stream);
}

/**
 * @brief Divide the source vector to the destination vector with the same
 * length
 * TODO: divide by zero check
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src Source device-side vector
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Div_Kernel(ValueT *d_dst, T *d_src, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_dst[idx] /= d_src[idx];
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Div(ValueT *elements, T *values, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements, values,
      [] __host__ __device__(ValueT & element, T value) { element /= value; },
      length, target, stream);
}

/**
 * @brief Add the source vector to the destination vector with the same length
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_dst Destination device-side vector
 * @param[in] d_src1 Source device-side vector 1
 * @param[in] d_src2 Source device-side vector 2
 * @param[in] scale Scale factor
 * @param[in] length Vector length
 */
template <typename ValueT, typename T1, typename T2, typename T3,
          typename SizeT>
__global__ void Mad_Kernel(ValueT *d_dst, T1 *d_src1, T2 *d_src2, T3 scale,
                           SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_dst[idx] = d_src1[idx] * scale + d_src2[idx];
  }
}

template <typename ValueT, typename T1, typename T2, typename T3,
          typename SizeT>
cudaError_t Mad(ValueT *elements, T1 *src1s, T2 *src2s, T3 scale, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements, src1s, src2s,
      [scale] __host__ __device__(ValueT & element, T1 src1, T2 src2) {
        element = src1 * scale + src2;
      },
      length, target, stream);
}

}  // namespace oprtr

namespace util {

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
          unsigned int cudaHostRegisterFlag_in>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Set(
    Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
    SizeT length, Location target, cudaStream_t stream) {
  return ForEach(
      array_in, [] __host__ __device__(ValueT & element, ValueT_in element_in) {
        element = element_in;
      });
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
          unsigned int cudaHostRegisterFlag_in>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Add(
    Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
    SizeT length, Location target, cudaStream_t stream) {
  return ForEach(
      array_in, [] __host__ __device__(ValueT & element, ValueT_in element_in) {
        element += element_in;
      });
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
          unsigned int cudaHostRegisterFlag_in>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Minus(
    Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
    SizeT length, Location target, cudaStream_t stream) {
  return ForEach(
      array_in, [] __host__ __device__(ValueT & element, ValueT_in element_in) {
        element -= element_in;
      });
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
          unsigned int cudaHostRegisterFlag_in>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Mul(
    Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
    SizeT length, Location target, cudaStream_t stream) {
  return ForEach(
      array_in, [] __host__ __device__(ValueT & element, ValueT_in element_in) {
        element *= element_in;
      });
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename SizeT_in, typename ValueT_in, ArrayFlag FLAG_in,
          unsigned int cudaHostRegisterFlag_in>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Div(
    Array1D<SizeT_in, ValueT_in, FLAG_in, cudaHostRegisterFlag_in> &array_in,
    SizeT length, Location target, cudaStream_t stream) {
  return ForEach(
      array_in, [] __host__ __device__(ValueT & element, ValueT_in element_in) {
        element /= element_in;
      });
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename SizeT_in1, typename ValueT_in1, ArrayFlag FLAG_in1,
          unsigned int cudaHostRegisterFlag_in1, typename SizeT_in2,
          typename ValueT_in2, ArrayFlag FLAG_in2,
          unsigned int cudaHostRegisterFlag_in2, typename T>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Mad(
    Array1D<SizeT_in1, ValueT_in1, FLAG_in1, cudaHostRegisterFlag_in1>
        &array_in1,
    Array1D<SizeT_in2, ValueT_in2, FLAG_in2, cudaHostRegisterFlag_in2>
        &array_in2,
    T scale, SizeT length, Location target, cudaStream_t stream) {
  return ForEach(
      array_in1, array_in2,
      [scale] __host__ __device__(ValueT & element, ValueT_in1 element_in1,
                                  ValueT_in2 element_in2) {
        element = element_in1 * scale + element_in2;
      });
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
