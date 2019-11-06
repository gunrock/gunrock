// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sort_utils.cuh
 *
 * @brief Sorting utility functions including thrust sort, mgpu merge sort,
 * and CUB radix sort.
 */

/******************************************************************************
 * Sorting utility functions
 ******************************************************************************/

#pragma once
#include <cub/cub.cuh>
//#include <moderngpu.cuh>

namespace gunrock {
namespace util {

/**
 * \addtogroup PublicInterface
 * @{
 */
template <typename KeyType, typename ValueType>
cudaError_t CUBRadixSort(bool is_ascend, size_t num_elements, KeyType *d_key,
                         ValueType *d_value = NULL, KeyType *d_key_temp = NULL,
                         ValueType *d_value_temp = NULL,
                         void *d_temp_storage = NULL,
                         size_t temp_storage_bytes = 0,
                         cudaStream_t stream = 0) {
  cudaError_t retval = cudaSuccess;

  // KeyType *key = NULL;
  // ValueType *value = NULL;
  bool value_supplied = !(d_value == NULL);
  bool key_temp_supplied = !(d_key_temp == NULL);
  bool value_temp_supplied = !(d_value_temp == NULL);
  bool temp_storage_supplied = !(d_temp_storage == NULL);

  if (!key_temp_supplied) {
    if (util::GRError(
            (retval = cudaMalloc(&d_key_temp, sizeof(KeyType) * num_elements)),
            "CUBRadixSort key malloc failed", __FILE__, __LINE__))
      return retval;
  }
  if (value_supplied && !value_temp_supplied) {
    if (util::GRError((retval = cudaMalloc(&d_value_temp,
                                           sizeof(ValueType) * num_elements)),
                      "CUBRadixSort value malloc failed", __FILE__, __LINE__))
      return retval;
  }

  cub::DoubleBuffer<KeyType> key_buffer(d_key, d_key_temp);
  cub::DoubleBuffer<ValueType> value_buffer(d_value, d_value_temp);

  if (!temp_storage_supplied) {
    if (value_supplied && is_ascend) {
      if (util::GRError((retval = cub::DeviceRadixSort::SortPairs(
                             d_temp_storage, temp_storage_bytes, key_buffer,
                             value_buffer, num_elements)),
                        "cub::DeviceRadixSort::SortPairs failed", __FILE__,
                        __LINE__))
        return retval;

    } else if (value_supplied && !is_ascend) {
      if (util::GRError((retval = cub::DeviceRadixSort::SortPairsDescending(
                             d_temp_storage, temp_storage_bytes, key_buffer,
                             value_buffer, num_elements)),
                        "cub::DeviceRadixSort::SortPairsDescending failed",
                        __FILE__, __LINE__))
        return retval;

    } else if (!value_supplied && is_ascend) {
      if (util::GRError((retval = cub::DeviceRadixSort::SortKeys(
                             d_temp_storage, temp_storage_bytes, key_buffer,
                             num_elements)),
                        "cub::DeviceRadixSort::SortKeyss failed", __FILE__,
                        __LINE__))
        return retval;

    } else if (!value_supplied && !is_ascend) {
      if (util::GRError((retval = cub::DeviceRadixSort::SortKeysDescending(
                             d_temp_storage, temp_storage_bytes, key_buffer,
                             num_elements)),
                        "cub::DeviceRadixSort::SortKeysDescending failed",
                        __FILE__, __LINE__))
        return retval;
    }

    if (util::GRError(
            (retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)),
            "CUB RadixSort malloc d_temp_storage failed", __FILE__, __LINE__))
      return retval;
  }

  // void   *d_temp_storage = NULL;
  // size_t temp_storage_bytes = 0;
  if (value_supplied && is_ascend) {
    // Key Value Pair sort (according to keys)
    if (util::GRError(
            (retval = cub::DeviceRadixSort::SortPairs(
                 d_temp_storage, temp_storage_bytes, key_buffer, value_buffer,
                 num_elements, 0, sizeof(KeyType) * 8, stream)),
            "cub::DeviceRadixSort::SortPairs failed", __FILE__, __LINE__))
      return retval;

  } else if (value_supplied && !is_ascend) {
    if (util::GRError(
            (retval = cub::DeviceRadixSort::SortPairsDescending(
                 d_temp_storage, temp_storage_bytes, key_buffer, value_buffer,
                 num_elements, 0, sizeof(KeyType) * 8, stream)),
            "cub::DeviceRadixSort::SortPairsDescending failed", __FILE__,
            __LINE__))
      return retval;

  } else if (!value_supplied && is_ascend) {
    if (util::GRError((retval = cub::DeviceRadixSort::SortKeys(
                           d_temp_storage, temp_storage_bytes, key_buffer,
                           num_elements, 0, sizeof(KeyType) * 8, stream)),
                      "cub::DeviceRadixSort::SortKeys failed", __FILE__,
                      __LINE__))
      return retval;

  } else if (!value_supplied && !is_ascend) {
    if (util::GRError((retval = cub::DeviceRadixSort::SortKeysDescending(
                           d_temp_storage, temp_storage_bytes, key_buffer,
                           num_elements, 0, sizeof(KeyType) * 8)),
                      "cub::DeviceRadixSort::SortKeysDescending failed",
                      __FILE__, __LINE__))
      return retval;
  }

  if (d_key != key_buffer.Current()) {
    if (util::GRError(
            (retval = cudaMemcpyAsync(d_key, key_buffer.Current(),
                                      sizeof(KeyType) * num_elements,
                                      cudaMemcpyDeviceToDevice, stream)),
            "CUB RadixSort copy back keys failed", __FILE__, __LINE__))
      return retval;
  }

  if (value_supplied && d_value != value_buffer.Current()) {
    if (util::GRError(
            (retval = cudaMemcpyAsync(d_value, value_buffer.Current(),
                                      sizeof(ValueType) * num_elements,
                                      cudaMemcpyDeviceToDevice, stream)),
            "CUB RadixSort copy back values failed", __FILE__, __LINE__))
      return retval;
  }

  if (!temp_storage_supplied) {
    if (util::GRError((retval = cudaFree(d_temp_storage)),
                      "CUB Radixsort free d_temp_storage failed", __FILE__,
                      __LINE__))
      return retval;
  }

  if (!key_temp_supplied) {
    if (util::GRError((retval = cudaFree(d_key_temp)),
                      "CUB Radixsort free key failed", __FILE__, __LINE__))
      return retval;
  }

  if (value_supplied && !value_temp_supplied) {
    if (util::GRError((retval = cudaFree(d_value_temp)),
                      "CUB Radixsort free value failed", __FILE__, __LINE__))
      return retval;
  }

  return retval;
}

/*
 * @brief mordern gpu segmentated sort from indices
 * using SegSortKeysFromInidices(), SegSortPairsFromIndices()
 */
/*  template <
    typename SizeType,
    typename KeyType,
    typename ValType>
  cudaError_t SegSortFromIndices(
    mgpu::CudaContext &context,
    size_t            num_indices,
    SizeType          *d_indices,
    size_t            num_elements,
    KeyType           *d_key,
    ValType           *d_val = NULL)
  {

    cudaError_t retval = cudaSuccess;

    if (d_val)
    {
      mgpu::SegSortPairsFromIndices(
    d_key,
    d_val,
    num_elements,
    d_indices,
    num_indices,
    context);
    }
    else
    {
      mgpu::SegSortKeysFromIndices(
    d_key,
    num_elements,
    d_indices,
    num_indices,
    context);
    }

    return retval;
  }
*/
template <typename KeyT, typename ValueT, typename SizeT>
cudaError_t cubSortPairs(util::Array1D<uint64_t, char> &temp_space,
                         util::Array1D<SizeT, KeyT> &keys_in,
                         util::Array1D<SizeT, KeyT> &keys_out,
                         util::Array1D<SizeT, ValueT> &values_in,
                         util::Array1D<SizeT, ValueT> &values_out,
                         SizeT num_items, int begin_bit = 0,
                         int end_bit = sizeof(KeyT) * 8,
                         cudaStream_t stream = 0,
                         bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;
  cub::DoubleBuffer<KeyT> keys(
      const_cast<KeyT *>(keys_in.GetPointer(util::DEVICE)),
      keys_out.GetPointer(util::DEVICE));
  cub::DoubleBuffer<ValueT> values(
      const_cast<ValueT *>(values_in.GetPointer(util::DEVICE)),
      values_out.GetPointer(util::DEVICE));

  size_t request_bytes = 0;
  retval = cub::DispatchRadixSort<false, KeyT, ValueT, SizeT>::Dispatch(
      NULL, request_bytes, keys, values, num_items, begin_bit, end_bit, false,
      stream, debug_synchronous);
  if (retval) return retval;
  // util::PrintMsg("num_items = " + std::to_string(num_items)
  //    + ", request_bytes = " + std::to_string(request_bytes));

  retval = temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval = cub::DispatchRadixSort<false, KeyT, ValueT, SizeT>::Dispatch(
      temp_space.GetPointer(util::DEVICE), request_bytes, keys, values,
      num_items, begin_bit, end_bit, false, stream, debug_synchronous);
  if (retval) return retval;

  if (keys.Current() != keys_out.GetPointer(util::DEVICE)) {
    KeyT *keys_ = keys.Current();
    GUARD_CU(keys_out.ForAll(
        [keys_] __host__ __device__(KeyT * keys_o, const SizeT &pos) {
          keys_o[pos] = keys_[pos];
        },
        num_items, util::DEVICE, stream));
  }

  if (values.Current() != values_out.GetPointer(util::DEVICE)) {
    ValueT *values_ = values.Current();
    GUARD_CU(values_out.ForAll(
        [values_] __host__ __device__(ValueT * values_o, const SizeT &pos) {
          values_o[pos] = values_[pos];
        },
        num_items, util::DEVICE, stream));
  }
  return retval;
}

template <typename KeyT, typename ValueT, typename SizeT>
cudaError_t cubSortPairsDescending(util::Array1D<uint64_t, char> &temp_space,
                                   util::Array1D<SizeT, KeyT> &keys_in,
                                   util::Array1D<SizeT, KeyT> &keys_out,
                                   util::Array1D<SizeT, ValueT> &values_in,
                                   util::Array1D<SizeT, ValueT> &values_out,
                                   SizeT num_items, int begin_bit = 0,
                                   int end_bit = sizeof(KeyT) * 8,
                                   cudaStream_t stream = 0,
                                   bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;
  cub::DoubleBuffer<KeyT> keys(
      const_cast<KeyT *>(keys_in.GetPointer(util::DEVICE)),
      keys_out.GetPointer(util::DEVICE));
  cub::DoubleBuffer<ValueT> values(
      const_cast<ValueT *>(values_in.GetPointer(util::DEVICE)),
      values_out.GetPointer(util::DEVICE));

  size_t request_bytes = 0;
  retval = cub::DispatchRadixSort<true, KeyT, ValueT, SizeT>::Dispatch(
      NULL, request_bytes, keys, values, num_items, begin_bit, end_bit, false,
      stream, debug_synchronous);
  if (retval) return retval;

  retval = temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval = cub::DispatchRadixSort<true, KeyT, ValueT, SizeT>::Dispatch(
      temp_space.GetPointer(util::DEVICE), request_bytes, keys, values,
      num_items, begin_bit, end_bit, false, stream, debug_synchronous);
  if (retval) return retval;

  if (keys.Current() != keys_out.GetPointer(util::DEVICE)) {
    KeyT *keys_ = keys.Current();
    GUARD_CU(keys_out.ForAll(
        [keys_] __host__ __device__(KeyT * keys_o, const SizeT &pos) {
          keys_o[pos] = keys_[pos];
        },
        num_items, util::DEVICE, stream));
  }

  if (values.Current() != values_out.GetPointer(util::DEVICE)) {
    ValueT *values_ = values.Current();
    GUARD_CU(values_out.ForAll(
        [values_] __host__ __device__(ValueT * values_o, const SizeT &pos) {
          values_o[pos] = values_[pos];
        },
        num_items, util::DEVICE, stream));
  }

  return retval;
}

template <typename KeyT, typename ValueT, typename SizeT>
cudaError_t cubSegmentedSortPairs(
    util::Array1D<uint64_t, char> &temp_space,
    util::Array1D<SizeT, KeyT> &keys_in, util::Array1D<SizeT, KeyT> &keys_out,
    util::Array1D<SizeT, ValueT> &values_in,
    util::Array1D<SizeT, ValueT> &values_out, SizeT num_items,
    SizeT num_segments, util::Array1D<SizeT, SizeT> &seg_offsets,
    int begin_bit = 0, int end_bit = sizeof(KeyT) * 8, cudaStream_t stream = 0,
    bool debug_synchronous = false) {
  cudaError_t retval = cudaSuccess;

  cub::DoubleBuffer<KeyT> keys(
      const_cast<KeyT *>(keys_in.GetPointer(util::DEVICE)),
      keys_out.GetPointer(util::DEVICE));
  cub::DoubleBuffer<ValueT> values(
      const_cast<ValueT *>(values_in.GetPointer(util::DEVICE)),
      values_out.GetPointer(util::DEVICE));

  size_t request_bytes = 0;
  retval =
      cub::DispatchSegmentedRadixSort<false, KeyT, ValueT, SizeT *, SizeT>::
          Dispatch(NULL, request_bytes, keys, values, num_items, num_segments,
                   seg_offsets.GetPointer(util::DEVICE),
                   seg_offsets.GetPointer(util::DEVICE) + 1, begin_bit, end_bit,
                   false, stream, debug_synchronous);
  if (retval) return retval;

  retval = temp_space.EnsureSize_(request_bytes, util::DEVICE);
  if (retval) return retval;

  retval = cub::
      DispatchSegmentedRadixSort<false, KeyT, ValueT, SizeT *, SizeT>::Dispatch(
          temp_space.GetPointer(util::DEVICE), request_bytes, keys, values,
          num_items, num_segments, seg_offsets.GetPointer(util::DEVICE),
          seg_offsets.GetPointer(util::DEVICE) + 1, begin_bit, end_bit, false,
          stream, debug_synchronous);
  if (retval) return retval;

  return retval;
}

/** @} */

}  // namespace util
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
