// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * 1D_scalar.cuh
 *
 * @brief 1D array with scalar operations
 */

/******************************************************************************
 * 1D_scalar routines
 ******************************************************************************/
#pragma once

#include <gunrock/oprtr/1D_oprtr/for_all.cuh>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>

namespace gunrock {
namespace oprtr {

// TODO: The memset kernels are getting nasty.
// Need to use operator overload to rewrite most
// of these some day.

/**
 * \addtogroup PublicInterface
 * @{
 */

/**
 * @brief Memset a device vector.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Value we want to set
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Set_Kernel(ValueT *d_out, T value, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_out[idx] = value;
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Set(ValueT *elements, T value, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements,
      [value] __host__ __device__(ValueT & element) { element = value; },
      length, target, stream);
}

/*template <typename VertexId, typename SizeT, typename Value>
__global__ void MemsetAddEdgeValKernel(Coo<VertexId, Value> *d_out, VertexId
value, SizeT length)
{
   const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
    for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x;
         idx < length; idx += STRIDE)
    {
        d_out[idx].row += value;
        d_out[idx].col += value;
    }
}*/

/**
 * @brief Memset a device vector with the element's index in the vector
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] length Vector length
 * @param[in] scale The scale for indexing (1 by default)
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void SetIdx_Kernel(ValueT *d_out, T scale, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_out[idx] = idx * scale;
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t SetIdx(ValueT *elements,
                   T scale,  // = 1,
                   SizeT length, util::Location target = util::DEVICE,
                   cudaStream_t stream = 0) {
  return ForAll(elements,
                [scale] __host__ __device__(ValueT * elements, int pos) {
                  elements[pos] = pos * scale;
                },
                length, target, stream);
}

/**
 * @brief Add value to each element in a device vector.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Value we want to add to each element in the vector
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Add_Kernel(ValueT *d_out, T value, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_out[idx] += value;
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Add(ValueT *elements, T value, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements,
      [value] __host__ __device__(ValueT & element) { element += value; },
      length, target, stream);
}

/**
 * @brief Minus value to each element in a device vector.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Value we want to add to each element in the vector
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Minus_Kernel(ValueT *d_out, T value, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_out[idx] -= value;
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Minus(ValueT *elements, T value, SizeT length,
                  util::Location target = util::DEVICE,
                  cudaStream_t stream = 0) {
  return ForEach(
      elements,
      [value] __host__ __device__(ValueT & element) { element -= value; },
      length, target, stream);
}

/**
 * @brief Multiply each element in a device vector to a certain factor.
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Scale factor
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Mul_Kernel(ValueT *d_out, T value, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_out[idx] *= value;
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Mul(ValueT *elements, T value, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements,
      [value] __host__ __device__(ValueT & element) { element *= value; },
      length, target, stream);
}

/**
 * @brief Divide each element in a device vector to a certain factor.
 * TODO: divide by zero check
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Scale factor
 * @param[in] length Vector length
 */
template <typename ValueT, typename T, typename SizeT>
__global__ void Div_Kernel(ValueT *d_out, T value, SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    d_out[idx] /= value;
  }
}

template <typename ValueT, typename T, typename SizeT>
cudaError_t Div(ValueT *elements, T value, SizeT length,
                util::Location target = util::DEVICE, cudaStream_t stream = 0) {
  return ForEach(
      elements,
      [value] __host__ __device__(ValueT & element) { element /= value; },
      length, target, stream);
}

/**
 * @brief Compare an element to a comp, if equal, assign val to it
 *
 * @tparam T datatype of the vector.
 *
 * @param[in] d_out Device-side vector we need to process on
 * @param[in] value Scale factor
 * @param[in] length Vector length
 */
template <typename ValueT, typename CompareT, typename AssignT, typename SizeT>
__global__ void CAS_Kernel(ValueT *d_dst, CompareT compare, AssignT val,
                           SizeT length) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    if (d_dst[idx] == compare) d_dst[idx] = val;
  }
}

template <typename ValueT, typename CompareT, typename AssignT, typename SizeT>
__global__ void CAS_Kernel(ValueT *d_dst, CompareT compare, AssignT val,
                           SizeT *length_) {
  const SizeT STRIDE = (SizeT)gridDim.x * blockDim.x;
  SizeT length = length_[0];
  for (SizeT idx = ((SizeT)blockIdx.x * blockDim.x) + threadIdx.x; idx < length;
       idx += STRIDE) {
    if (d_dst[idx] == compare) d_dst[idx] = val;
  }
}
/** @} */

}  // namespace oprtr

namespace util {

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>
    &Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::operator=(T val) {
  GRError(Set(val), std::string(name) + " Set() failed.", __FILE__, __LINE__);
  return (*this);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>
    &Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::operator+=(T val) {
  GRError(Add(val), std::string(name) + " Add() failed.", __FILE__, __LINE__);
  return (*this);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>
    &Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::operator-=(T val) {
  GRError(Minus(val), std::string(name) + " Minus() failed.", __FILE__,
          __LINE__);
  return (*this);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>
    &Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::operator*=(T val) {
  GRError(Minus(val), std::string(name) + " Mul() failed.", __FILE__, __LINE__);
  return (*this);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>
    &Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::operator/=(T val) {
  GRError(Minus(val), std::string(name) + " Div() failed.", __FILE__, __LINE__);
  return (*this);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Set(
    T value,
    SizeT length,  // = PreDefinedValues<typename ArrayT::SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  return ForEach(
      [value] __host__ __device__(ValueT & element) { element = value; },
      length, target, stream);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::SetIdx(
    T scale,       // = 1,
    SizeT length,  // = PreDefinedValues<typename ArrayT::SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  return ForAll(
      [scale] __host__ __device__(ValueT * elements, SizeT pos) {
        elements[pos] = pos * scale;
      },
      length, target, stream);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Add(
    T value,
    SizeT length,  // = PreDefinedValues<typename ArrayT::SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  return ForEach(
      [value] __host__ __device__(ValueT & element) { element += value; },
      length, target, stream);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Minus(
    T value,
    SizeT length,  // = PreDefinedValues<typename ArrayT::SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  return ForEach(
      [value] __host__ __device__(ValueT & element) { element -= value; },
      length, target, stream);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Mul(
    T value,
    SizeT length,  // = PreDefinedValues<typename ArrayT::SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  return ForEach(
      [value] __host__ __device__(ValueT & element) { element *= value; },
      length, target, stream);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename T>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Div(
    T value,
    SizeT length,  // = PreDefinedValues<typename ArrayT::SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  return ForEach(
      [value] __host__ __device__(ValueT & element) { element /= value; },
      length, target, stream);
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename CompareT, typename AssignT>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::CAS(
    CompareT compare, AssignT assign,
    SizeT length,  // = PreDefinedValues<typename ArrayT::SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  // typedef typename ArrayT::ValueT ValueT;
  return ForEach(
      [compare, assign] __host__ __device__(ValueT & element) {
        if (element == compare) element = assign;
      },
      length, target, stream);
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
