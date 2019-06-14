// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sort.cuh
 *
 * @brief Sorting of 1D array
 */

#pragma once
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sort.h>
#include <gunrock/util/array_utils.cuh>
#include <gunrock/util/sort_omp.cuh>
#include <gunrock/oprtr/1D_oprtr/for_each.cuh>

namespace gunrock {
namespace util {

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename CompareLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Sort(
    CompareLambda compare,  // = [] __host__ __device__
    //(const ValueT &a, const ValueT &b){
    //    return a < b;
    //},
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    SizeT offset,         // = 0,
    Location target,      // = util::LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  cudaError_t retval = cudaSuccess;
  if (length == PreDefinedValues<SizeT>::InvalidValue) length = this->GetSize();
  if (target == util::LOCATION_DEFAULT) target = this->setted | this->allocated;

  if (target & DEVICE) {
    thrust::sort(thrust::cuda::par.on(stream),
                 this->GetPointer(util::DEVICE) + offset,
                 this->GetPointer(util::DEVICE) + (offset + length), compare);
    if (target & HOST) {
      if (retval = this->Move(DEVICE, HOST, length, offset, stream))
        return retval;
    }
  } else {
    if (retval = util::omp_sort((*this) + offset, length, compare))
      return retval;
  }
  return retval;
}

template <typename ValueT, typename AuxValueT>
struct tValuePair {
  ValueT org_value;
  AuxValueT aux_value;
};

template <typename SizeT, typename ValueT, ArrayFlag FLAG,
          unsigned int cudaHostRegisterFlag>
template <typename ArrayT, typename CompareLambda>
cudaError_t Array1D<SizeT, ValueT, FLAG, cudaHostRegisterFlag>::Sort_by_Key(
    ArrayT &array_in,
    CompareLambda compare,  // = [] __host__ __device__
    //(const ValueT &a, const ValueT &b){
    //    return a < b;
    //},
    SizeT length,         // = PreDefinedValues<SizeT>::InvalidValue,
    SizeT offset,         // = 0,
    Location target,      // = LOCATION_DEFAULT,
    cudaStream_t stream)  // = 0)
{
  typedef typename ArrayT::ValueT AuxValueT;
  typedef tValuePair<ValueT, AuxValueT> tValueT;

  cudaError_t retval = cudaSuccess;
  if (target & DEVICE) {
    thrust::device_ptr<ValueT> org_values(this->GetPointer(DEVICE) + offset);
    thrust::device_ptr<AuxValueT> aux_values(array_in.GetPointer(DEVICE) +
                                             offset);
    thrust::device_vector<ValueT> org_vector(org_values, org_values + length);
    thrust::device_vector<AuxValueT> aux_vector(aux_values,
                                                aux_values + length);
    thrust::sort_by_key(thrust::cuda::par.on(stream), org_vector.begin(),
                        org_vector.end(), aux_vector.begin(), compare);
    if (target & HOST) {
      if (retval = this->Move(DEVICE, HOST, length, offset, stream))
        return retval;
      if (retval = array_in.Move(DEVICE, HOST, length, offset, stream))
        return retval;
    }
  } else {
    // perpare temp array for sorting
    Array1D<SizeT, tValueT, FLAG, cudaHostRegisterFlag> temp_array;
    temp_array.SetName("Array1D::Sort::temp_array");
    if (retval = temp_array.Allocate(length, target)) return retval;
    if (retval = oprtr::ForEach(
            temp_array + 0, (*this) + offset, array_in + offset,
            [] __host__ __device__(tValueT & temp_value, const ValueT &value,
                                   const AuxValueT &value_in) {
              temp_value.org_value = value;
              temp_value.aux_value = value_in;
            },
            length, target, stream))
      return retval;

    // the actual sorting
    if (retval = omp_sort(temp_array + 0, length,
                          [compare](const tValueT &a, const tValueT &b) {
                            return compare(a.org_value, b.org_value);
                          }))
      return retval;

    // Copy back to orginal space
    if (retval = oprtr::ForEach(
            temp_array + 0, (*this) + offset, array_in + offset,
            [] __host__ __device__(const tValueT &temp_value, ValueT &value,
                                   AuxValueT &value_out) {
              value = temp_value.org_value;
              value_out = temp_value.aux_value;
            },
            length, target, stream))
      return retval;

    // Deallocate temp array
    if (retval = temp_array.Release()) return retval;
  }
  return retval;
}

}  // namespace util
}  // namespace gunrock
