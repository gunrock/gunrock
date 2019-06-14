// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * for_all.cuh
 *
 * @brief Simple "for all" operations
 */

#pragma once

#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace oprtr {

#define FORALL_BLOCKSIZE 256
#define FORALL_GRIDSIZE 256
/*template <
    typename T,
    typename SizeT,
    typename ApplyLambda>
__global__ void ForAll_Kernel(
    T          *d_array,
    ApplyLambda apply,
    SizeT       length)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        apply(d_array, i);
        i += STRIDE;
    }
}*/

template <typename ArrayT, typename SizeT, typename ApplyLambda>
__global__ void ForAll_Kernel(ArrayT array, ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    apply(array + 0, i);
    i += STRIDE;
  }
}

/*template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename ApplyLambda>
__global__ void ForAll_Kernel(
    T_in       *d_ins,
    T_out      *d_outs,
    ApplyLambda apply,
    SizeT       length)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        apply(d_ins, d_outs, i);
        i += STRIDE;
    }
}*/

template <typename ArrayT_out, typename ArrayT_in, typename SizeT,
          typename ApplyLambda>
__global__ void ForAll_Kernel(ArrayT_out array_out, ArrayT_in array_in,
                              ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  // printf("(%d, %d) length = %d\n", blockIdx.x, threadIdx.x, length);
  while (i < length) {
    // printf("Applying %d\n", i);
    apply(array_out + 0, array_in + 0, i);
    i += STRIDE;
  }
}

template <typename ArrayT_out, typename ArrayT_in1, typename ArrayT_in2,
          typename SizeT, typename ApplyLambda>
__global__ void ForAll_Kernel(ArrayT_out array_out, ArrayT_in1 array_in1,
                              ArrayT_in2 array_in2, ApplyLambda apply,
                              SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  // printf("(%d, %d) length = %d\n", blockIdx.x, threadIdx.x, length);
  while (i < length) {
    // printf("Applying %d\n", i);
    apply(array_out + 0, array_in1 + 0, array_in2 + 0, i);
    i += STRIDE;
  }
}

/*template <
    typename T,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForAllCond_Kernel(
    T          *d_array,
    CondLambda  cond,
    ApplyLambda apply,
    SizeT       length)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        if (cond(d_array, i))
            apply(d_array, i);
        i += STRIDE;
    }
}*/

template <typename ArrayT, typename SizeT, typename CondLambda,
          typename ApplyLambda>
__global__ void ForAllCond_Kernel(ArrayT array, CondLambda cond,
                                  ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    if (cond(array + 0, i)) apply(array + 0, i);
    i += STRIDE;
  }
}

/*template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForAllCond_Kernel(
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
        if (cond(d_ins, d_outs, i))
            apply(d_ins, d_outs, i);
        i += STRIDE;
    }
}*/

template <typename ArrayT_out, typename ArrayT_in, typename SizeT,
          typename CondLambda, typename ApplyLambda>
__global__ void ForAllCond_Kernel(ArrayT_out array_out, ArrayT_in array_in,
                                  CondLambda cond, ApplyLambda apply,
                                  SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    if (cond(array_out + 0, array_in + 0, i))
      apply(array_out + 0, array_in + 0, i);
    i += STRIDE;
  }
}

template <typename ArrayT_out, typename ArrayT_in1, typename ArrayT_in2,
          typename SizeT, typename CondLambda, typename ApplyLambda>
__global__ void ForAllCond_Kernel(ArrayT_out array_out, ArrayT_in1 array_in1,
                                  ArrayT_in2 array_in2, CondLambda cond,
                                  ApplyLambda apply, SizeT length) {
  // typedef typename ArrayT_in::SizeT SizeT;
  const SizeT STRIDE = (SizeT)blockDim.x * gridDim.x;
  SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
  while (i < length) {
    if (cond(array_out + 0, array_in1 + 0, array_in2 + 0, i))
      apply(array_out + 0, array_in1 + 0, array_in2 + 0, i);
    i += STRIDE;
  }
}

template <typename T, typename SizeT, typename ApplyLambda>
cudaError_t ForAll(T *elements, ApplyLambda apply, SizeT length,
                   util::Location target = util::DEVICE,
                   cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) apply(elements, i);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForAll_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        elements, apply, length);
  }
  return retval;
}

template <typename T_out, typename T_in, typename SizeT, typename ApplyLambda>
cudaError_t ForAll(T_out *elements_out, T_in *elements_in, ApplyLambda apply,
                   SizeT length, util::Location target = util::HOST,
                   cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) apply(elements_out, elements_in, i);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForAll_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        elements_out, elements_in, apply, length);
  }
  return retval;
}

template <typename T_out, typename T_in1, typename T_in2, typename SizeT,
          typename ApplyLambda>
cudaError_t ForAll(T_out *elements_out, T_in1 *elements_in1,
                   T_in2 *elements_in2, ApplyLambda apply, SizeT length,
                   util::Location target = util::HOST,
                   cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      apply(elements_out, elements_in1, elements_in2, i);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForAll_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        elements_out, elements_in1, elements_in2, apply, length);
  }
  return retval;
}

template <typename T, typename SizeT, typename CondLambda, typename ApplyLambda>
cudaError_t ForAllCond(T *elements, CondLambda cond, ApplyLambda apply,
                       SizeT length, util::Location target = util::DEVICE,
                       cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond(elements, i)) apply(elements, i);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForAllCond_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        elements, cond, apply, length);
  }
  return retval;
}

template <typename T_out, typename T_in, typename SizeT, typename CondLambda,
          typename ApplyLambda>
cudaError_t ForAllCond(T_out *elements_out, T_in *elements_in, CondLambda cond,
                       ApplyLambda apply, SizeT length,
                       util::Location target = util::DEVICE,
                       cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond(elements_out, elements_in, i))
        apply(elements_out, elements_in, i);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForAllCond_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        elements_out, elements_in, cond, apply, length);
  }
  return retval;
}

template <typename T_out, typename T_in1, typename T_in2, typename SizeT,
          typename CondLambda, typename ApplyLambda>
cudaError_t ForAllCond(T_out *elements_out, T_in1 *elements_in1,
                       T_in2 *elements_in2, CondLambda cond, ApplyLambda apply,
                       SizeT length, util::Location target = util::DEVICE,
                       cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;
  if ((target & util::HOST) == util::HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++)
      if (cond(elements_out, elements_in1, elements_in2, i))
        apply(elements_out, elements_in1, elements_in2, i);
  }

  if ((target & util::DEVICE) == util::DEVICE) {
    ForAllCond_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        elements_out, elements_in1, elements_in2, cond, apply, length);
  }
  return retval;
}

}  // namespace oprtr

namespace util {

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForAll(
    // ArrayT       array,
    ApplyLambda apply,
    SizeT length,         //= PreDefinedValues<SizeT>::InvalidValue,
    Location target,      // = util::LOCATION_DEFAULT,
    cudaStream_t stream,  // = 0,
    int grid_size,        // = util::PreDefinedValues<int>::InvalidValue
    int block_size)       // = util::PreDefinedValues<int>::InvalidValue
{
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == LOCATION_DEFAULT) target = this->setted | this->allocated;

  if ((target & HOST) == HOST) {
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) apply((*this) + 0, i);
  }

  if ((target & DEVICE) == DEVICE) {
    if (!util::isValid(grid_size)) grid_size = FORALL_GRIDSIZE;
    if (!util::isValid(block_size)) block_size = FORALL_BLOCKSIZE;
    oprtr::ForAll_Kernel<<<grid_size, block_size, 0, stream>>>((*this), apply,
                                                               length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForAll(
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
    for (SizeT i = 0; i < length; i++) apply((*this) + 0, array_in + 0, i);
  }

  if ((target & DEVICE) == DEVICE) {
    // printf("Launch kernel, length = %d\n", length);
    oprtr::ForAll_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        (*this), array_in, apply, length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in1, typename ArrayT_in2, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForAll(
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
// util::PrintMsg("Launching on HOST, length = " + std::to_string(length));
#pragma omp parallel for
    for (SizeT i = 0; i < length; i++) {
      // util::PrintMsg(std::to_string(i) + " " + std::to_string((*this)[i]));
      apply((*this) + 0, array_in1 + 0, array_in2 + 0, i);
    }
  }

  if ((target & DEVICE) == DEVICE) {
    // util::PrintMsg("Launching on DEVICE, length = " +
    // std::to_string(length));
    oprtr::ForAll_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        (*this), array_in1, array_in2, apply, length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename CondLambda, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForAllCond(
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
      if (cond((*this) + 0, i)) apply((*this) + 0, i);
  }

  if ((target & DEVICE) == DEVICE) {
    oprtr::ForAllCond_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        (*this), cond, apply, length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in, typename CondLambda, typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForAllCond(
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
      if (cond((*this) + 0, array_in + 0, i))
        apply((*this) + 0, array_in + 0, i);
  }

  if ((target & DEVICE) == DEVICE) {
    oprtr::ForAllCond_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
        (*this), array_in, cond, apply, length);
  }
  return retval;
}

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT_in1, typename ArrayT_in2, typename CondLambda,
          typename ApplyLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::ForAllCond(
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
      if (cond((*this) + 0, array_in1 + 0, array_in2 + 0, i))
        apply((*this) + 0, array_in1 + 0, array_in2 + 0, i);
  }

  if ((target & DEVICE) == DEVICE) {
    oprtr::ForAllCond_Kernel<<<FORALL_GRIDSIZE, FORALL_BLOCKSIZE, 0, stream>>>(
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
