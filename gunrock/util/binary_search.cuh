// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * binary search.cuh
 *
 * @brief Binary search routine
 */

#pragma once

namespace gunrock {
namespace util {

template <typename T1, typename ArrayT, typename SizeT,
    typename CompareLambda>
__host__ __device__
SizeT BinarySearch(
    const T1& element_to_find,
    const ArrayT& elements,
    SizeT left_index,
    SizeT right_index,
    CompareLambda comp) // strictly less
{
    SizeT center_index = 0;
    do {
        if (right_index - left_index <= 1)
        {
            if (comp(elements[right_index], element_to_find))
                return right_index + 1;
            if (comp(elements[left_index], element_to_find))
                return right_index;
            return left_index;
        } else {
            center_index = ((long long)left_index + (long long) right_index) >> 1;
            if (comp(elements[center_index], element_to_find))
            {
                left_index = center_index + 1;
            } else {
                right_index = center_index - 1;
            }
#ifndef __CUDA_ARCH__
            /*PrintMsg("Find " + std::to_string(element_to_find) +
                " Pos " + std::to_string(center_index) +
                " -> [" + std::to_string(left_index) +
                ", " + std::to_string(right_index) + "]");*/
#endif
        }
    } while (true);
    //return 0;
}

} //namespace util
} //namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
