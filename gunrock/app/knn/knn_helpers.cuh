// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * knn_helpers.cuh
 *
 * @brief Helper functions for KNN algorithm.
 */

#pragma once

#include <map>
#include <cmath>
#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {
namespace knn {

template <typename ValueT>
__device__ __host__ ValueT _sqrt(const ValueT &a) {
    return (ValueT) sqrtf((float)a);
}

/**
 * @brief Compute euclidean distance
 * @param dim Number of dimensions (2D, 3D ... ND)
 * @param points Points array to get the x, y, z...
 * @param p1 and p2 points to be compared
 * info \return distance value
 */
template<typename SizeT, typename ValueT, typename PointT>
__device__ __host__
ValueT euclidean_distance(const SizeT dim, 
    util::Array1D<SizeT, ValueT> points, 
    PointT p1, PointT p2) {

    // Get dimensional of labels
    ValueT result = (ValueT) 0;
    // p1 = (x_1, x_2, ..., x_dim)
    // p2 = (y_1, y_2, ..., y_dim)
    for (int i=0; i<dim; ++i){
        //(x_i - y_i)^2
        ValueT diff = points[p1 * dim + i] - points[p2 * dim + i];
        result += diff*diff;
    }
    return _sqrt(result);
}

}  // namespace knn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
