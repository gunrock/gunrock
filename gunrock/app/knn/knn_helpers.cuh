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
    return sqrt(a);
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
                    //assert(step < blockDim.x);
                    if (new_dist[threadIdx.x] > new_dist[step]){
                        auto tmp = new_dist[step];
                        new_dist[step] = new_dist[threadIdx.x];
                        new_dist[threadIdx.x] = tmp;
                        auto tmp2 = new_keys[step];
                        new_keys[step] = new_keys[threadIdx.x];
                        new_keys[threadIdx.x] = tmp2;
                    }
                }else{
                    //assert(step < blockDim.x);
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
 * @param points Points array to get the x, y, z...
 * @param p1 and p2 points to be compared
 * info \return distance value
 */
template<typename SizeT, typename ValueT, typename PointT>
__device__ __host__
ValueT euclidean_distance(const SizeT dim, const SizeT N, 
    ValueT* points, 
    PointT p1, PointT p2, bool transpose) {

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
        result += diff*diff;
    }
    return _sqrt(result);
}

template<typename SizeT, typename ValueT, typename PointT>
__device__ __host__
ValueT euclidean_distance(const SizeT dim, const SizeT N, 
    ValueT* points, const PointT p1, ValueT* sh_point, bool transpose) {

    // Get dimensional of labels
    ValueT result = (ValueT) 0;
    // p1 = (x_1, x_2, ..., x_dim)
    // p2 = (y_1, y_2, ..., y_dim)
    for (int i=0; i<dim; ++i){
        //(x_i - y_i)^2
        ValueT diff = (ValueT)0;
        if (! transpose){
            diff = points[p1 * dim + i] - sh_point[i];
        }else{
            diff = points[i * N + p1] - sh_point[i];
        }
        result += diff*diff;
    }
    return _sqrt(result);
}
template<typename SizeT, typename ValueT, typename PointT>
__device__ __host__
ValueT euclidean_distance(const SizeT dim, const SizeT N, 
    ValueT* b_point, PointT p1, ValueT* points, PointT p2){

    // Get dimensional of labels
    ValueT result = (ValueT) 0;
    // p1 = (x_1, x_2, ..., x_dim)
    // p2 = (y_1, y_2, ..., y_dim)
    for (int i=0; i<dim; ++i){
        //(x_i - y_i)^2
        ValueT diff = (ValueT)0;
        diff = b_point[i * (blockDim.x+1) + p1] - points[i * N + p2];
        result += diff*diff;
    }
    return _sqrt(result);
}
template<typename SizeT, typename ValueT, typename PointT>
__device__
ValueT euclidean_distance(const SizeT dim, 
        ValueT* b_points, PointT p1, 
        ValueT* sh_points, PointT p2){

    // Get dimensional of labels
    ValueT result = (ValueT) 0;
    // p1 = (x_1, x_2, ..., x_dim)
    // p2 = (y_1, y_2, ..., y_dim)
    for (int i=0; i<dim; ++i){
        //(x_i - y_i)^2
        ValueT diff = b_points[i * blockDim.x + p1] - sh_points[i * blockDim.x + p2];
        result += diff*diff;
    }
    return _sqrt(result);
}

template<typename SizeT, typename ValueT>
__global__
void init(ValueT arraySH, ValueT arrayG, SizeT cols, SizeT row0, SizeT num_rows){
    for (int i = 0; i<num_rows; ++i){
        arraySH[threadIdx.x * (num_rows+1) + i] = arrayG[(row0 + i) * cols + threadIdx.x];
    }
}
 
template<typename SizeT, typename ValueT, typename PointT>
__device__
ValueT euclidean_distance(const SizeT dim, 
        ValueT* b_points, PointT p1, 
        ValueT* sh_point){

    // Get dimensional of labels
    ValueT result = (ValueT) 0;
    // p1 = (x_1, x_2, ..., x_dim)
    // p2 = (y_1, y_2, ..., y_dim)
    for (int i=0; i<dim; ++i){
        //(x_i - y_i)^2
        ValueT diff = b_points[i * (blockDim.x+1) + p1] - sh_point[i];
        result += diff*diff;
    }
    return _sqrt(result);
}

template<typename SizeT, typename ValueT>
__global__ 
void euclidean_distanceDim1024(const SizeT dim, const SizeT N, const SizeT k,
    ValueT* points, const SizeT p1, const SizeT p2, ValueT* distance) {

    __shared__ float values[2];

 //   printf("threads (%d, %d), block (%d, %d)\n",
 //           blockDim.x, blockDim.y, gridDim.x, gridDim.y);

    if (threadIdx.x < dim){
        float diff = (float)(points[(threadIdx.x * N) + p1] - points[(threadIdx.x * N) + p2]);
        values[threadIdx.x] = diff*diff; 
    }else{
        values[threadIdx.x] = (float)0;
    }
    __syncthreads();
/*
    if (threadIdx.x == 0){
        for (int i = 0; i<dim; ++i){
            printf("(%.lf - %.lf)^2 = values[%d] = %.f\n", points[i * N + p1], points[i * N + p2], i, values[i]);
        }
    }
*/

    for (int i = 1; i < dim; i*=2){
        float diff = (float)0;
        if (threadIdx.x + i < dim){
            diff = values[threadIdx.x + i];
        }
        __syncthreads();
        values[threadIdx.x] += diff;
        __syncthreads();
    }

    __syncthreads();

    if (threadIdx.x == 0){
        distance[p1 * (k+1) + k] = _sqrt(values[0]);
        //printf("dist (%d, %d) = %.f\n", p1, p2, distance[p1*(k+1) + k]);
    }

    return;
}
}  // namespace knn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
