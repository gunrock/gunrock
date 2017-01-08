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
#include <moderngpu.cuh>

namespace gunrock {
namespace util {

/**
* \addtogroup PublicInterface
* @{
*/
template <typename KeyType, typename ValueType>
cudaError_t CUBRadixSort(
    bool      is_ascend,
    size_t    num_elements,
    KeyType   *d_key,
    ValueType *d_value = NULL,
    KeyType   *d_key_temp = NULL,
    ValueType *d_value_temp = NULL,
    void      *d_temp_storage = NULL,
    size_t     temp_storage_bytes = 0,
    cudaStream_t stream = 0)
{

    cudaError_t retval = cudaSuccess;

    //KeyType *key = NULL;
    //ValueType *value = NULL;
    bool value_supplied        = !(d_value        == NULL);
    bool key_temp_supplied     = !(d_key_temp     == NULL);
    bool value_temp_supplied   = !(d_value_temp   == NULL);
    bool temp_storage_supplied = !(d_temp_storage == NULL);

    if (!key_temp_supplied)
    {
        if (util::GRError((retval = cudaMalloc(
            &d_key_temp, sizeof(KeyType) * num_elements)),
            "CUBRadixSort key malloc failed",
            __FILE__, __LINE__)) return retval;
    }
    if (value_supplied && !value_temp_supplied)
    {
        if (util::GRError((retval = cudaMalloc(
            &d_value_temp, sizeof(ValueType) * num_elements)),
            "CUBRadixSort value malloc failed",
            __FILE__, __LINE__)) return retval;
    }

    cub::DoubleBuffer<KeyType>   key_buffer  (d_key  , d_key_temp  );
    cub::DoubleBuffer<ValueType> value_buffer(d_value, d_value_temp);

    if (!temp_storage_supplied)
    {
        if (value_supplied && is_ascend)
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortPairs(
                d_temp_storage,
                temp_storage_bytes,
                key_buffer,
                value_buffer,
                num_elements)),
                "cub::DeviceRadixSort::SortPairs failed",
                __FILE__, __LINE__)) return retval;

        } else if (value_supplied && !is_ascend)
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortPairsDescending(
                d_temp_storage,
                temp_storage_bytes,
                key_buffer,
                value_buffer,
                num_elements)),
                "cub::DeviceRadixSort::SortPairsDescending failed",
                __FILE__, __LINE__)) return retval;

        } else if (!value_supplied && is_ascend)
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortKeys(
                d_temp_storage,
                temp_storage_bytes,
                key_buffer,
                num_elements)),
                "cub::DeviceRadixSort::SortKeyss failed",
                __FILE__, __LINE__)) return retval;

        } else if (!value_supplied && !is_ascend)
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortKeysDescending(
                d_temp_storage,
                temp_storage_bytes,
                key_buffer,
                num_elements)),
                "cub::DeviceRadixSort::SortKeysDescending failed",
                __FILE__, __LINE__)) return retval;
        }

        if (util::GRError((retval = cudaMalloc(
            &d_temp_storage, temp_storage_bytes)),
            "CUB RadixSort malloc d_temp_storage failed",
            __FILE__, __LINE__)) return retval;
    }

    //void   *d_temp_storage = NULL;
    //size_t temp_storage_bytes = 0;
    if (value_supplied && is_ascend)
    {
        //Key Value Pair sort (according to keys)
        if (util::GRError((retval = cub::DeviceRadixSort::SortPairs(
            d_temp_storage,
            temp_storage_bytes,
            key_buffer,
            value_buffer,
            num_elements,
            0, sizeof(KeyType) * 8,
            stream)),
            "cub::DeviceRadixSort::SortPairs failed",
            __FILE__, __LINE__)) return retval;

    } else if (value_supplied && !is_ascend)
    {
        if (util::GRError((retval = cub::DeviceRadixSort::SortPairsDescending(
            d_temp_storage,
            temp_storage_bytes,
            key_buffer,
            value_buffer,
            num_elements,
            0, sizeof(KeyType) * 8,
            stream)),
            "cub::DeviceRadixSort::SortPairsDescending failed",
            __FILE__, __LINE__)) return retval;

    } else if (!value_supplied && is_ascend)
    {
        if (util::GRError((retval = cub::DeviceRadixSort::SortKeys(
            d_temp_storage,
            temp_storage_bytes,
            key_buffer,
            num_elements,
            0, sizeof(KeyType) * 8,
            stream)),
            "cub::DeviceRadixSort::SortKeys failed",
            __FILE__, __LINE__)) return retval;

    } else if (!value_supplied && !is_ascend)
    {
        if (util::GRError((retval = cub::DeviceRadixSort::SortKeysDescending(
            d_temp_storage,
            temp_storage_bytes,
            key_buffer,
            num_elements,
            0, sizeof(KeyType) * 8)),
            "cub::DeviceRadixSort::SortKeysDescending failed",
            __FILE__, __LINE__)) return retval;
    }

    if (d_key != key_buffer.Current())
    {
        if (util::GRError((retval = cudaMemcpyAsync(
            d_key,
            key_buffer.Current(),
            sizeof(KeyType)*num_elements,
            cudaMemcpyDeviceToDevice, stream)),
            "CUB RadixSort copy back keys failed",
            __FILE__, __LINE__)) return retval;
    }

    if (value_supplied && d_value != value_buffer.Current())
    {
        if (util::GRError((retval = cudaMemcpyAsync(
            d_value,
            value_buffer.Current(),
            sizeof(ValueType)*num_elements,
            cudaMemcpyDeviceToDevice, stream)),
            "CUB RadixSort copy back values failed",
            __FILE__, __LINE__)) return retval;
    }

    if (!temp_storage_supplied)
    {
        if (util::GRError((retval = cudaFree(d_temp_storage)),
            "CUB Radixsort free d_temp_storage failed",
            __FILE__, __LINE__)) return retval;
    }

    if (!key_temp_supplied)
    {
        if (util::GRError((retval = cudaFree(d_key_temp)),
            "CUB Radixsort free key failed",
            __FILE__, __LINE__)) return retval;
    }

    if (value_supplied && !value_temp_supplied)
    {
        if (util::GRError((retval = cudaFree(d_value_temp)),
            "CUB Radixsort free value failed",
            __FILE__, __LINE__)) return retval;
    }

    return retval;
  }


  /*
   * @brief mordern gpu segmentated sort from indices
   * using SegSortKeysFromInidices(), SegSortPairsFromIndices()
   */
  template <
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

  /** @} */

} //util
} //gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:

