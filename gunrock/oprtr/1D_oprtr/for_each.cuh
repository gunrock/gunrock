// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * for_each.cuh
 *
 * @brief Simple element-wise "for each" operations
 */

#pragma once

#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace oprtr {

/*template <
    typename T,
    typename SizeT,
    typename ApplyLambda>
__global__ void ForEach_Kernel(
    T          *d_array,
    ApplyLambda apply,
    SizeT       length)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        apply(d_array[i]);
        i += STRIDE;
    }
}*/

template <typename ArrayT, typename SizeT, typename ApplyLambda>
__global__ void ForEach_Kernel(ArrayT array, ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    apply(array[i]);
    i += STRIDE;
  }
}

/*template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename ApplyLambda>
__global__ void ForEach_Kernel(
    T_in       *d_ins,
    T_out      *d_outs,
    ApplyLambda apply,
    SizeT       length)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        apply(d_ins[i], d_outs[i]);
        i += STRIDE;
    }
}*/

/*template <
    typename ArrayT_out,
    typename ArrayT_in,
    typename SizeT,
    typename ApplyLambda>
__global__ void ForEach_Kernel(
    ArrayT_out  array_out,
    ArrayT_in   array_in,
    ApplyLambda apply,
    SizeT       length)
{
    //typedef typename ArrayT_in::SizeT SizeT;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    //printf("(%d, %d) length = %d\n", blockIdx.x, threadIdx.x, length);
    while (i < length)
    {
        //printf("Applying %d\n", i);
        apply(array_out[i], array_in[i]);
        i += STRIDE;
    }
}*/

template <typename ArrayT_out, typename ArrayT_in, typename SizeT,
          typename ApplyLambda>
__global__ void ForEach_Kernel(ArrayT_out array_out, ArrayT_in array_in,
                               ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  // printf("(%d, %d) length = %d\n", blockIdx.x, threadIdx.x, length);
  while (i < length) {
    // printf("Applying %d\n", i);
    apply(array_out[i], array_in[i]);
    i += STRIDE;
  }
}

template <typename ArrayT_out, typename ArrayT_in1, typename ArrayT_in2,
          typename SizeT, typename ApplyLambda>
__global__ void ForEach_Kernel(ArrayT_out array_out, ArrayT_in1 array_in1,
                               ArrayT_in2 array_in2, ApplyLambda apply,
                               SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  // printf("(%d, %d) length = %d\n", blockIdx.x, threadIdx.x, length);
  while (i < length) {
    // printf("Applying %d\n", i);
    apply(array_out[i], array_in1[i], array_in2[i]);
    i += STRIDE;
  }
}

/*template <
    typename T,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForEachCond_Kernel(
    T          *d_array,
    CondLambda  cond,
    ApplyLambda apply,
    SizeT       length)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        if (cond(d_array[i]))
            apply(d_array[i]);
        i += STRIDE;
    }
}*/

template <typename ArrayT, typename SizeT, typename CondLambda,
          typename ApplyLambda>
__global__ void ForEachCond_Kernel(ArrayT array, CondLambda cond,
                                   ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    if (cond(array[i])) apply(array[i]);
    i += STRIDE;
  }
}

/*template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForEachCond_Kernel(
    T_in       *d_ins,
    T_out      *d_outs,
    CondLambda  cond,
    ApplyLambda apply,
    SizeT       length)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        if (cond(d_ins[i], d_outs[i]))
            apply(d_ins[i], d_outs[i]);
        i += STRIDE;
    }
}*/

template <typename ArrayT_out, typename ArrayT_in, typename SizeT,
          typename CondLambda, typename ApplyLambda>
__global__ void ForEachCond_Kernel(ArrayT_out array_out, ArrayT_in array_in,
                                   CondLambda cond, ApplyLambda apply,
                                   SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    if (cond(array_out[i], array_in[i])) apply(array_out[i], array_in[i]);
    i += STRIDE;
  }
}

template <typename ArrayT_out, typename ArrayT_in1, typename ArrayT_in2,
          typename SizeT, typename CondLambda, typename ApplyLambda>
__global__ void ForEachCond_Kernel(ArrayT_out array_out, ArrayT_in1 array_in1,
                                   ArrayT_in2 array_in2, CondLambda cond,
                                   ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    if (cond(array_out[i], array_in1[i], array_in2[i]))
      apply(array_out[i], array_in1[i], array_in2[i]);
    i += STRIDE;
  }
}

template <typename T, typename SizeT, typename ApplyLambda>
cudaError_t ForEach(T *elements, ApplyLambda apply, SizeT length,
                    util::Location target = util::DEVICE,
                    cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) apply(elements[i]);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForEach_Kernel<<<256, 256, 0, stream>>>(elements, apply, length);
  }
  return retval;
}

template <typename T_out, typename T_in, typename SizeT, typename ApplyLambda>
cudaError_t ForEach(T_out *elements_out, T_in *elements_in, ApplyLambda apply,
                    SizeT length, util::Location target = util::DEVICE,
                    cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) apply(elements_out[i], elements_in[i]);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForEach_Kernel<<<256, 256, 0, stream>>>(elements_out, elements_in, apply,
                                            length);
  }
  return retval;
}

template <typename T_out, typename T_in1, typename T_in2, typename SizeT,
          typename ApplyLambda>
cudaError_t ForEach(T_out *elements_out, T_in1 *elements_in1,
                    T_in2 *elements_in2, ApplyLambda apply, SizeT length,
                    util::Location target = util::DEVICE,
                    cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      apply(elements_out[i], elements_in1[i], elements_in2[i]);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForEach_Kernel<<<256, 256, 0, stream>>>(elements_out, elements_in1,
                                            elements_in2, apply, length);
  }
  return retval;
}

template <typename T, typename SizeT, typename CondLambda, typename ApplyLambda>
cudaError_t ForEachCond(T *elements, CondLambda cond, ApplyLambda apply,
                        SizeT length, util::Location target = util::DEVICE,
                        cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond(elements[i])) apply(elements[i]);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForEachCond_Kernel<<<256, 256, 0, stream>>>(elements, cond, apply, length);
  }
  return retval;
}

template <typename T_out, typename T_in, typename SizeT, typename CondLambda,
          typename ApplyLambda>
cudaError_t ForEachCond(T_out *elements_out, T_in *elements_in, CondLambda cond,
                        ApplyLambda apply, SizeT length,
                        util::Location target = util::DEVICE,
                        cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond(elements_out[i], elements_in[i]))
        apply(elements_out[i], elements_in[i]);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForEachCond_Kernel<<<256, 256, 0, stream>>>(elements_out, elements_in, cond,
                                                apply, length);
  }
  return retval;
}

template <typename T_out, typename T_in1, typename T_in2, typename SizeT,
          typename CondLambda, typename ApplyLambda>
cudaError_t ForEachCond(T_out *elements_out, T_in1 *elements_in1,
                        T_in2 *elements_in2, CondLambda cond, ApplyLambda apply,
                        SizeT length, util::Location target = util::DEVICE,
                        cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond(elements_out[i], elements_in1[i], elements_in2[i]))
        apply(elements_out[i], elements_in1[i], elements_in2[i]);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForEachCond_Kernel<<<256, 256, 0, stream>>>(
        elements_out, elements_in1, elements_in2, cond, apply, length);
  }
  return retval;
}

}  // namespace oprtr

namespace util {

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForEach(
    // ArrayT       array,
    ApplyLambda apply,
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    Location target,      // = util::LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == util::LOCATION_DEFAULT) target = this->setted | this->allocated;

  if ((target & HOST) == HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) apply((*this)[i]);
  }

  if ((target & DEVICE) == DEVICE) {
    oprtr::ForEach_Kernel<<<256, 256, 0, stream>>>((*this), apply, length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForEach(
    ArrayT_in &array_in,
    // ArrayT_out   array_out,
    ApplyLambda apply,
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  // typedef typename ArrayT_in::SizeT SizeT;
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == util::LOCATION_DEFAULT) target = this->setted | this->allocated;

  if ((target & HOST) == HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) apply((*this)[i], array_in[i]);
  }

  if ((target & DEVICE) == DEVICE) {
    // printf("Launch kernel, length = %d\n", length);
    oprtr::ForEach_Kernel<<<256, 256, 0, stream>>>((*this), array_in, apply,
                                                   length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in1, typename ArrayT_in2, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForEach(
    ArrayT_in1 &array_in1, ArrayT_in2 &array_in2,
    // ArrayT_out   array_out,
    ApplyLambda apply,
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  // typedef typename ArrayT_in::SizeT SizeT;
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == util::LOCATION_DEFAULT) target = this->setted | this->allocated;

  if ((target & HOST) == HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      apply((*this)[i], array_in1[i], array_in2[i]);
  }

  if ((target & DEVICE) == DEVICE) {
    // printf("Launch kernel, length = %d\n", length);
    oprtr::ForEach_Kernel<<<256, 256, 0, stream>>>((*this), array_in1,
                                                   array_in2, apply, length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename CondLambda, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForEachCond(
    // ArrayT       array,
    CondLambda cond, ApplyLambda apply,
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  // typedef typename ArrayT::SizeT SizeT;
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == LOCATION_DEFAULT) target = this->setted | this->allocated;

  if ((target & HOST) == HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond((*this)[i])) apply((*this)[i]);
  }

  if ((target & DEVICE) == DEVICE) {
    oprtr::ForEachCond_Kernel<<<256, 256, 0, stream>>>((*this), cond, apply,
                                                       length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in, typename CondLambda, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForEachCond(
    ArrayT_in &array_in,
    // ArrayT_out   array_out,
    CondLambda cond, ApplyLambda apply,
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  // typedef typename ArrayT_in::SizeT SizeT;
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == util::LOCATION_DEFAULT) target = this->setted | this->allocated;

  if ((target & HOST) == HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond((*this)[i], array_in[i])) apply((*this)[i], array_in[i]);
  }

  if ((target & DEVICE) == DEVICE) {
    oprtr::ForEachCond_Kernel<<<256, 256, 0, stream>>>((*this), array_in, cond,
                                                       apply, length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in1, typename ArrayT_in2, typename CondLambda,
          typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForEachCond(
    ArrayT_in1 &array_in1, ArrayT_in2 &array_in2,
    // ArrayT_out   array_out,
    CondLambda cond, ApplyLambda apply,
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  // typedef typename ArrayT_in::SizeT SizeT;
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == util::LOCATION_DEFAULT) target = this->setted | this->allocated;

  if ((target & HOST) == HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond((*this)[i], array_in1[i], array_in2[i]))
        apply((*this)[i], array_in1[i], array_in2[i]);
  }

  if ((target & DEVICE) == DEVICE) {
    oprtr::ForEachCond_Kernel<<<256, 256, 0, stream>>>(
        (*this), array_in1, array_in2, cond, apply, length);
  }
  return retval;
}

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
