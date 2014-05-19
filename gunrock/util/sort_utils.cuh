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
// thrust sort
//#include <thrust/sort.h>
//#include <thrust/functional.h>
// mgpu merge_sort
//#include <moderngpu.cuh>
// cub radix_sort
#include <cub/cub/cub.cuh>

namespace gunrock {
namespace util {

template <typename KeyType, typename ValueType>
cudaError_t CUBRadixSort(
        bool        is_ascend, 
        size_t      num_elements,
        KeyType     *d_key,
        ValueType   *d_value = NULL)
{
    cudaError_t retval = cudaSuccess;

    KeyType *key = NULL;
    ValueType *value = NULL;
    if (util::GRError((retval = cudaMalloc(&key, sizeof(KeyType)*num_elements)), 
        "CUBRadixSort key malloc failed", __FILE__, __LINE__)) return retval;
    if (d_value)
    {
        if (util::GRError((retval = cudaMalloc(&value, sizeof(ValueType)*num_elements)),
            "CUBRadixSort value malloc failed", __FILE__, __LINE__)) return retval;
    }
    cub::DoubleBuffer<KeyType> key_buffer(d_key, key);
    cub::DoubleBuffer<ValueType> value_buffer(d_value, value);

    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    if (d_value) 
    {
        // key value pair sort (according to keys)
        if (is_ascend) 
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortPairs(d_temp_storage, 
                temp_storage_bytes, key_buffer, value_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortPairs failed", __FILE__, __LINE__)) return retval; 
            if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)), 
                "CUB RadixSort malloc d_temp_storage failed", __FILE__, __LINE__)) return retval;
            if (util::GRError((retval = cub::DeviceRadixSort::SortPairs(d_temp_storage, 
                temp_storage_bytes, key_buffer, value_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortPairs failed", __FILE__, __LINE__)) return retval;
        }
        else
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, key_buffer, value_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortPairsDescending failed", __FILE__, __LINE__)) return retval; 
            if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)), 
                "CUB RadixSort malloc d_temp_storage failed", __FILE__, __LINE__)) return retval;
            if (util::GRError((retval = cub::DeviceRadixSort::SortPairsDescending(
                d_temp_storage, temp_storage_bytes, key_buffer, value_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortPairsDescending failed", __FILE__, __LINE__)) return retval; 
        }
    }
    else
    {
        if (is_ascend)
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortKeys(
                d_temp_storage, temp_storage_bytes, key_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortKeyss failed", __FILE__, __LINE__)) return retval; 
            if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)), 
                "CUB RadixSort malloc d_temp_storage failed", __FILE__, __LINE__)) return retval;
            if (util::GRError((retval = cub::DeviceRadixSort::SortKeys(
                d_temp_storage, temp_storage_bytes, key_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortKeys failed", __FILE__, __LINE__)) return retval;
        }
        else
        {
            if (util::GRError((retval = cub::DeviceRadixSort::SortKeysDescending(
                d_temp_storage, temp_storage_bytes, key_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortKeysDescending failed", __FILE__, __LINE__)) return retval; 
            if (util::GRError((retval = cudaMalloc(&d_temp_storage, temp_storage_bytes)), 
                "CUB RadixSort malloc d_temp_storage failed", __FILE__, __LINE__)) return retval;
            if (util::GRError((retval = cub::DeviceRadixSort::SortKeysDescending(
                d_temp_storage, temp_storage_bytes, key_buffer, num_elements)), 
                "CUB::DeviceRadixSort::SortKeysDescending failed", __FILE__, __LINE__)) return retval;
        }
    }
    if (util::GRError((retval = cudaMemcpy(d_key, key_buffer.Current(), 
        sizeof(KeyType)*num_elements, cudaMemcpyDeviceToDevice)), 
        "CUB RadixSort copy back keys failed", __FILE__, __LINE__)) return retval;
    if (util::GRError((retval = cudaMemcpy(d_value, value_buffer.Current(), 
        sizeof(ValueType)*num_elements, cudaMemcpyDeviceToDevice)), 
        "CUB RadixSort copy back values failed", __FILE__, __LINE__)) return retval;

    if (util::GRError((retval = cudaFree(d_temp_storage)), 
        "CUB Radixsort free d_temp_storage failed", __FILE__, __LINE__)) return retval;
    if (util::GRError((retval = cudaFree(key)), 
        "CUB Radixsort free key failed", __FILE__, __LINE__)) return retval;
    if (util::GRError((retval = cudaFree(value)), 
        "CUB Radixsort free value failed", __FILE__, __LINE__)) return retval;

    return retval;
}




    /** @} */

} // util
} // gunrock
