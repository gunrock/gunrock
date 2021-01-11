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
    return (ValueT) sqrt((double)a);
}

template <typename ValueT, typename SizeT>
__device__ void bitonic_sort(ValueT* new_dist, SizeT* new_keys, int length){
    // Bitonic sort on new_dist array:
    for (int offset = 2; offset <= length; offset *= 2){
        #pragma unroll
        for (int p = offset/2; p > 0; p /= 2){
            int step = threadIdx.x ^ p;
            if (step > threadIdx.x){
                if ((threadIdx.x & offset) == 0){
                    if (new_dist[threadIdx.x] > new_dist[step]){
                        auto tmp = new_dist[step];
                        new_dist[step] = new_dist[threadIdx.x];
                        new_dist[threadIdx.x] = tmp;
                        auto tmp2 = new_keys[step];
                        new_keys[step] = new_keys[threadIdx.x];
                        new_keys[threadIdx.x] = tmp2;
                    }
                }else{
                    if (new_dist[threadIdx.x] < new_dist[step]){
                        auto tmp = new_dist[step];
                        new_dist[step] = new_dist[threadIdx.x];
                        new_dist[threadIdx.x] = tmp;
                        auto tmp2 = new_keys[step];
                        new_keys[step] = new_keys[threadIdx.x];
                        new_keys[threadIdx.x] = tmp2;
                    }
                }
            }
            __syncthreads();
        }
    }
}

__device__ void acquire_semaphore(int* lock, int i){
    while (atomicCAS(&lock[i], 0, 1) != 0);
    __threadfence();
}

__device__ void release_semaphore(int* lock, int i){
    __threadfence();
    lock[i] = 0;
}
/**
 * @brief Compute euclidean distance
 * @param dim Number of dimensions (2D, 3D ... ND)
 * @param N Number of points
 * @param points Points array to get the x, y, z...
 * @param p1 and p2 points to be compared
 * @param transpose is true if points array is transposed
 * info \return distance value
 * Use in operator knn_general_op, knn_half_op, 
 */
template<typename SizeT, typename ValueT, typename PointT>
__device__ __host__
ValueT euclidean_distance(const SizeT dim, const SizeT N, 
    ValueT* points, PointT p1, PointT p2, bool transpose) {

    // Get dimensional of labels
    ValueT result = (ValueT) 0;
    // p1 = (x_1, x_2, ..., x_dim)
    // p2 = (y_1, y_2, ..., y_dim)
    for (int i=0; i<dim; ++i){
        //(x_i - y_i)^2
        ValueT diff = (ValueT)0;
        if (! transpose){
            diff = points[p1 * dim + i] - points[p2 * dim + i];
        }else{
            diff = points[i * N + p1] - points[i * N + p2];
        }
        assert(std::abs(diff) < std::abs(util::PreDefinedValues<ValueT>::MaxValue/diff));
        assert(result < (util::PreDefinedValues<ValueT>::MaxValue - (diff*diff)));
        result += diff*diff;
    }
    return _sqrt(result);
}
// Use in operator knn_shared_not_transpose_op, knn_shared_transpose_op
template<typename SizeT, typename ValueT, typename PointT>
__device__
ValueT euclidean_distance(const SizeT dim, ValueT* b_points, PointT p1, 
        ValueT* sh_point){

    // Get dimensional of labels
    ValueT result = (ValueT) 0;
    // p1 = (x_1, x_2, ..., x_dim)
    // p2 = (y_1, y_2, ..., y_dim)
    for (int i=0; i<dim; ++i){
        //(x_i - y_i)^2
        ValueT diff = b_points[i * (blockDim.x+1) + p1] - sh_point[i];
        assert(std::abs(diff) < std::abs(util::PreDefinedValues<ValueT>::MaxValue/diff));
        assert(result < (util::PreDefinedValues<ValueT>::MaxValue - (diff*diff)));
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
