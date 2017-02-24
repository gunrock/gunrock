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
namespace util {

template <
    typename T,
    typename SizeT,
    typename ApplyLambda>
__global__ void ForAll_Kernel(
    T          *d_array,
    SizeT       length,
    ApplyLambda apply)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        apply(d_array, i);
        i += STRIDE;
    }
}

template <
    typename ArrayT,
    typename ApplyLambda>
__global__ void ForAll_Kernel(
    ArrayT      array,
    typename ArrayT::SizeT length,
    ApplyLambda apply)
{
    typedef typename ArrayT::SizeT SizeT;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        apply(array + 0, i);
        i += STRIDE;
    }
}

template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename ApplyLambda>
__global__ void ForAll_Kernel(
    T_in       *d_ins,
    T_out      *d_outs,
    SizeT       length,
    ApplyLambda apply)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        apply(d_ins, d_outs, i);
        i += STRIDE;
    }
}

template <
    typename ArrayT_in,
    typename ArrayT_out,
    typename ApplyLambda>
__global__ void ForAll_Kernel(
    ArrayT_in   array_in,
    ArrayT_out  array_out,
    typename ArrayT_in::SizeT length,
    ApplyLambda apply)
{
    typedef typename ArrayT_in::SizeT SizeT;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    //printf("(%d, %d) length = %d\n", blockIdx.x, threadIdx.x, length);
    while (i < length)
    {
        //printf("Applying %d\n", i);
        apply(array_in + 0, array_out + 0, i);
        i += STRIDE;
    }
}

template <
    typename T,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForAllCond_Kernel(
    T          *d_array,
    SizeT       length,
    CondLambda  cond,
    ApplyLambda apply)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        if (cond(d_array, i))
            apply(d_array, i);
        i += STRIDE;
    }
}

template <
    typename ArrayT,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForAllCond_Kernel(
    ArrayT      array,
    typename ArrayT::SizeT length,
    CondLambda  cond,
    ApplyLambda apply)
{
    typedef typename ArrayT::SizeT SizeT;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        if (cond(array + 0, i))
            apply(array + 0, i);
        i += STRIDE;
    }
}

template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForAllCond_Kernel(
    T_in       *d_ins,
    T_out      *d_outs,
    SizeT       length,
    CondLambda  cond,
    ApplyLambda apply)
{
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        if (cond(d_ins, d_outs, i))
            apply(d_ins, d_outs, i);
        i += STRIDE;
    }
}

template <
    typename ArrayT_in,
    typename ArrayT_out,
    typename CondLambda,
    typename ApplyLambda>
__global__ void ForAllCond_Kernel(
    ArrayT_in   array_in,
    ArrayT_out  array_out,
    typename ArrayT_in::SizeT length,
    CondLambda  cond,
    ApplyLambda apply)
{
    typedef typename ArrayT_in::SizeT SizeT;
    const SizeT STRIDE = (SizeT) blockDim.x * gridDim.x;
    SizeT i = (SizeT)blockDim.x * blockIdx.x + threadIdx.x;
    while (i < length)
    {
        if (cond(array_in + 0, array_out + 0, i))
            apply(array_in + 0, array_out + 0, i);
        i += STRIDE;
    }
}

template <
    typename T,
    typename SizeT,
    typename ApplyLambda>
cudaError_t ForAll(
    T           *elements,
    SizeT        length,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            apply(elements, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        ForAll_Kernel
            <<<256, 256, 0, stream>>>(
            elements, length, apply);
    }
    return retval;
}

template <
    typename ArrayT,
    typename ApplyLambda>
cudaError_t ForAll(
    ArrayT       array,
    typename ArrayT::SizeT length,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    typedef typename ArrayT::SizeT SizeT;
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            apply(array + 0, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        ForAll_Kernel
            <<<256, 256, 0, stream>>>(
            array, length, apply);
    }
    return retval;
}

template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename ApplyLambda>
cudaError_t ForAll(
    T_in        *elements_in,
    T_out       *elements_out,
    SizeT        length,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            apply(elements_in, elements_out, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        ForAll_Kernel
            <<<256, 256, 0, stream>>>(
            elements_in, elements_out, length, apply);
    }
    return retval;
}

template <
    typename ArrayT_in,
    typename ArrayT_out,
    typename ApplyLambda>
cudaError_t ForAll(
    ArrayT_in    array_in,
    ArrayT_out   array_out,
    typename ArrayT_in::SizeT length,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    typedef typename ArrayT_in::SizeT SizeT;
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            apply(array_in + 0, array_out + 0, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        //printf("Launch kernel, length = %d\n", length);
        ForAll_Kernel
            <<<256, 256, 0, stream>>>(
            array_in, array_out, length, apply);
    }
    return retval;
}

template <
    typename T,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
cudaError_t ForAllCond(
    T           *elements,
    SizeT        length,
    CondLambda   cond,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            if (cond(elements, i))
                apply(elements, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        ForAllCond_Kernel
            <<<256, 256, 0, stream>>>(
            elements, length, cond, apply);
    }
    return retval;
}

template <
    typename ArrayT,
    typename CondLambda,
    typename ApplyLambda>
cudaError_t ForAllCond(
    ArrayT       array,
    typename ArrayT::SizeT length,
    CondLambda   cond,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    typedef typename ArrayT::SizeT SizeT;
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            if (cond(array + 0, i))
                apply(array + 0, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        ForAllCond_Kernel
            <<<256, 256, 0, stream>>>(
            array, length, cond, apply);
    }
    return retval;
}

template <
    typename T_in,
    typename T_out,
    typename SizeT,
    typename CondLambda,
    typename ApplyLambda>
cudaError_t ForAllCond(
    T_in        *elements_in,
    T_out       *elements_out,
    SizeT        length,
    CondLambda   cond,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            if (cond(elements_in, elements_out, i))
                apply(elements_in, elements_out, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        ForAllCond_Kernel
            <<<256, 256, 0, stream>>>(
            elements_in, elements_out, length, cond, apply);
    }
    return retval;
}

template <
    typename ArrayT_in,
    typename ArrayT_out,
    typename CondLambda,
    typename ApplyLambda>
cudaError_t ForAllCond(
    ArrayT_in    array_in,
    ArrayT_out   array_out,
    typename ArrayT_in::SizeT length,
    CondLambda   cond,
    ApplyLambda  apply,
    Location     target = HOST,
    cudaStream_t stream = 0)
{
    typedef typename ArrayT_in::SizeT SizeT;
    cudaError_t retval = cudaSuccess;
    if ((target & HOST) == HOST)
    {
        #pragma omp parallel for
        for (SizeT i=0; i<length; i++)
            if (cond(array_in + 0, array_out + 0, i))
                apply(array_in + 0, array_out + 0, i);
    }

    if ((target & DEVICE) == DEVICE)
    {
        ForAllCond_Kernel
            <<<256, 256, 0, stream>>>(
            array_in, array_out, length, cond, apply);
    }
    return retval;
}

} // namespace util
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
