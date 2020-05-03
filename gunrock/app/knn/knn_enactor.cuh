// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * knn_enactor.cuh
 *
 * @brief knn Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <gunrock/app/knn/knn_problem.cuh>
#include <gunrock/app/knn/knn_helpers.cuh>
#include <gunrock/util/scan_device.cuh>
#include <gunrock/util/sort_device.cuh>

#include <gunrock/oprtr/1D_oprtr/for.cuh>
#include <gunrock/oprtr/oprtr.cuh>

#include <cstdio>

#include <cub/cub.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_radix_sort.cuh>

//do not remove debug
//#define KNN_ENACTOR_DEBUG
#ifdef KNN_ENACTOR_DEBUG
    #define debug(a...) printf(a)
#else
    #define debug(a...)
#endif

#define debug(a...)

namespace gunrock {
namespace app {
namespace knn {

/**
 * @brief Speciflying parameters for knn Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));

  return retval;
}

/**
 * @brief defination of knn iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct knnIterationLoop : public IterationLoopBase<EnactorT, Use_FullQ | Push> {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CsrT CsrT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;

  typedef IterationLoopBase<EnactorT, Use_FullQ | Push> BaseIterationLoop;

  knnIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of knn, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // --
    // Alias variables

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];

    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];

    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &oprtr_parameters = enactor_slice.oprtr_parameters;
    auto &retval = enactor_stats.retval;
    
    // K-Nearest Neighbors
    auto &keys_out = data_slice.knns;
    auto &distance_out = data_slice.distance_out;

    // Number of KNNs
    auto k = data_slice.k;
    // Number of points
    auto num_points = data_slice.num_points;
    // Dimension of labels
    auto dim = data_slice.dim;

    // List of points
    auto &points = data_slice.points;
    auto &sem = data_slice.sem;
    bool transpose = this->enactor->problem->transpose;

    cudaStream_t stream = oprtr_parameters.stream;
    auto target = util::DEVICE;

    if (transpose)
        debug("euclidean_distance will be use transpose version\n");
    else
        debug("euclidean distance wont be use transpose version\n");
   
    auto USE_SHARED_MEM = this->enactor->problem->use_shared_mem;
    int block_size = this->enactor->problem->block_size;
    int grid_size = this->enactor->problem->grid_size;
    int data_size = this->enactor->problem->data_size;
    int points_size = this->enactor->problem->points_size;
    int dist_size = this->enactor->problem->dist_size;
    int keys_size = this->enactor->problem->keys_size;
    //int shared_point_size = this->enactor->problem->shared_point_size;
    int shared_mem_size = this->enactor->problem->shared_mem_size;

    /* Operators */
    auto knn_general_op = 
    [num_points, k, dim, points, keys_out, transpose] 
    __device__ (ValueT* d, const SizeT &src, char* shared){
        ValueT* new_dist = (ValueT*)shared;
        int* dist_key = (int*)(shared + (blockDim.x * 8));
        dist_key[threadIdx.x] = src;
        for (SizeT i = 0; i<num_points; ++i){
            new_dist[threadIdx.x] = (ValueT)0;
            if (src == i || src >= num_points) {
                new_dist[threadIdx.x] = util::PreDefinedValues<ValueT>::MaxValue;
            } else {
                new_dist[threadIdx.x] = euclidean_distance(dim, num_points, points.GetPointer(util::DEVICE), src, i, transpose);
            }

            if (src < num_points && new_dist[threadIdx.x] < d[src * k + k - 1]) {
                // new element is smaller than the largest in distance array for "src" row
                SizeT current = k - 1;
                #pragma unroll
                for (; current > 0; --current){
                    SizeT one_before = current - 1;
                    if (new_dist[threadIdx.x] >= d[src * k + one_before]){
                        d[src * k + current] = new_dist[threadIdx.x];
                        keys_out[src * k + current] = i;
                        break;
                    } else {
                        d[src * k + current] = d[src * k + one_before];
                        keys_out[src * k + current] = keys_out[src * k + one_before];
                    }
                }
                if (current == (SizeT)0){
                    d[src * k] = new_dist[threadIdx.x];
                    keys_out[src * k] = i;
                }

            }
        }
    };
    
    auto knn_half_op = 
    [num_points, k, dim, points, keys_out, transpose, sem] 
    __device__ (ValueT* d, const SizeT &src, char* shared){
        ValueT* new_dist = (ValueT*)shared;
        SizeT* new_keys = (SizeT*)(shared + (blockDim.x * 8));
        
        int offset = (src/(blockDim.x*gridDim.x))*blockDim.x*gridDim.x;
        for (SizeT i0 = offset; i0<num_points; ++i0){
            SizeT i = offset + ((i0 + blockIdx.x)%(num_points-offset));

            if (i != src && src < num_points) {
                new_dist[threadIdx.x] = euclidean_distance(dim, num_points, points.GetPointer(util::DEVICE), src, i, transpose);
                new_keys[threadIdx.x] = src;
            }else{
                new_dist[threadIdx.x] = util::PreDefinedValues<ValueT>::MaxValue;
                new_keys[threadIdx.x] = util::PreDefinedValues<SizeT>::InvalidValue;
            }

            acquire_semaphore(sem.GetPointer(util::DEVICE), src);

            if (src < num_points && new_dist[threadIdx.x] < *((volatile ValueT*)(&d[src * k + k - 1]))) {
                SizeT current = k - 1;
                #pragma unroll
                for (; current > 0; --current){
                    SizeT one_before = current - 1;
                    if (new_dist[threadIdx.x] >= *((volatile ValueT*)(&d[src * k + one_before]))){
                        *((volatile ValueT*)(&d[src * k + current])) = new_dist[threadIdx.x];
                        *((volatile int*)(&keys_out[src * k + current])) = i;
                        break;
                    } else {
                        *((volatile ValueT*)(&d[src * k + current])) = *((volatile ValueT*)(&d[src * k + one_before]));
                        *((volatile int*)(&keys_out[src * k + current])) = *((volatile int*)(&keys_out[src * k + one_before]));
                    }
                }
                if (current == (SizeT)0){
                    *((volatile ValueT*)(&d[src * k])) = new_dist[threadIdx.x];
                    *((volatile int*)(&keys_out[src * k])) = i;
                }
            }

            release_semaphore(sem.GetPointer(util::DEVICE), src);
            __syncthreads();

            if (i >= offset+(blockDim.x*gridDim.x) && i < num_points){
 
                __syncthreads();
            
                // Bitonic sort on new_dist array:
                bitonic_sort(new_dist, new_keys, blockDim.x);
               
                __syncthreads();

                // Close semaphore for i row
                if (threadIdx.x == 0){
                    acquire_semaphore(sem.GetPointer(util::DEVICE), i);
                }

                // Find k smallest elements and merge them together to one array.
                if (threadIdx.x == 0){
                    int y = 0;
                    #pragma unroll
                    for (int x = 0; x + y < k;){
                        if (new_dist[y] <= *((volatile ValueT*)(&d[i * k + x]))){
                            ++y;
                        }else{
                            ++x;
                        }
                    }
                    #pragma unroll
                    for (int j = 0; y + j < k; ++j){
                        new_dist[y + j] = *((volatile ValueT*)(&d[i * k + j]));
                        new_keys[y + j] = *((volatile int*)(&keys_out[i * k + j]));
                    }
                }

                __syncthreads();

                // Bitonic sort on new_dist array:
                bitonic_sort(new_dist, new_keys, blockDim.x);
              
                __syncthreads();
                #pragma unroll
                for (int j = threadIdx.x; j<k; j += blockDim.x){
                    *((volatile ValueT*)(&d[i * k + j])) = new_dist[j];
                    *((volatile int*)(&keys_out[i * k + j])) = new_keys[j];
                }

                __syncthreads();
                
                if (threadIdx.x == 0){
                    release_semaphore(sem.GetPointer(util::DEVICE), i);
                }
            }
        }
    };

    auto knn_shared_not_transpose_op = 
    [num_points, k, dim, points, keys_out, data_size, points_size, dist_size, keys_size] 
    __device__ (ValueT* d, const SizeT &src, char* shared_mem){

        // Get pointers to shared memory arrays
        ValueT* dist = (ValueT*) (shared_mem);
        ValueT* b_sh_points = (ValueT*) (shared_mem + dist_size);
        int* keys = (int*) (shared_mem + dist_size + points_size);
        ValueT* sh_point = (ValueT*) (shared_mem + dist_size + points_size + keys_size);

        __shared__ SizeT firstPoint;
        if (threadIdx.x == 0){
            firstPoint = src;
        }
        __syncthreads();

        // Copying to shared memory
        if (dim%2 == 0){
            #pragma unroll
            for (SizeT j = threadIdx.x; j < (blockDim.x * dim)/2; j += blockDim.x){
                if constexpr(sizeof(ValueT) == 8){
                    // ValueT == double
                    reinterpret_cast<double2*>(b_sh_points)[j] = reinterpret_cast<double2*>(points + firstPoint*dim)[j];
                }else{
                    // ValueT == float
                    reinterpret_cast<float2*>(b_sh_points)[j] = reinterpret_cast<float2*>(points + firstPoint*dim)[j];
                }
            }
        }else{
            #pragma unroll
            for (SizeT j = threadIdx.x; j < blockDim.x * dim; j += blockDim.x){
                b_sh_points[j] = points[firstPoint*dim + j];
            }
        }

        __syncthreads();
        
        // Initializations of basic points
        // 7217ms 
        ValueT array[100];

        // Copying shared memory to registers
        #pragma unroll
        for (SizeT j = 0; j < dim; ++j){
            array[j] = b_sh_points[threadIdx.x * dim + j];
        }
        __syncthreads();

        // Transpose to shared memory
        #pragma unroll
        for (SizeT j = 0; j<dim; ++j){
            b_sh_points[j * (blockDim.x+1) + threadIdx.x] = array[j];
        }

        // Initializations of dist and keys
        #pragma unroll
        for (int i = 0; i < k; ++i){
            int idx = i * (blockDim.x+1) + threadIdx.x;
            dist[idx] = util::PreDefinedValues<ValueT>::MaxValue;
            //keys[idx] = util::PreDefinedValues<int>::InvalidValue;
        }

        __syncthreads();

        for (SizeT i = 0; i<num_points; ++i){
            
            // Initialization of shared points (points [i...i*blocDim.x] in sh_points)
            // Proceeding points[[0..dim] * num_points + i];
            if (dim%2 == 0){
                #pragma unroll
                for (SizeT j=threadIdx.x; j<dim/2; j+=blockDim.x){
                    // Doing better with fetching int4 data
                    if constexpr(sizeof(ValueT) == 8){
                        // ValueT == double
                        reinterpret_cast<double2*>(sh_point)[j] = reinterpret_cast<double2*>(points + (i * dim))[j];
                    }else{
                        // ValueT == float
                        reinterpret_cast<float2*>(sh_point)[j] = reinterpret_cast<float2*>(points + (i * dim))[j];
                    }
                }
            }else{
                #pragma unroll
                for (SizeT j=threadIdx.x; j<dim; j+=blockDim.x){
                    // Doing better with fetching int4 data
                    sh_point[j] = points[i * dim + j];
                }
            }
            __syncthreads();
            
            ValueT new_dist = 0;
            if (src == i || src >= num_points) {
                new_dist = util::PreDefinedValues<ValueT>::MaxValue;
            } else {
                new_dist = euclidean_distance(dim, b_sh_points, (int)threadIdx.x, sh_point);
            } 
            if (new_dist < dist[((k-1) * (blockDim.x + 1)) + threadIdx.x]) {
                // new element is larger than the largest in distance array for "src" row
                // new_dist < dist[(k-1) * blockDim.x + threadIdx.x]
                SizeT current = k-1;
                #pragma unroll
                for (; current > 0; --current){
                    SizeT one_before = current-1;
                    if (new_dist >= dist[(one_before * (blockDim.x + 1)) + threadIdx.x]){
                        dist[(current * (blockDim.x + 1)) + threadIdx.x] = new_dist;
                        keys[(current * (blockDim.x + 1)) + threadIdx.x] = i;
                        break;
                    } else {
                        dist[(current * (blockDim.x + 1)) + threadIdx.x] = dist[(one_before * (blockDim.x + 1)) + threadIdx.x];
                        keys[(current * (blockDim.x + 1)) + threadIdx.x] = keys[(one_before * (blockDim.x + 1)) + threadIdx.x];
                    }
                }
                if (current == (SizeT)0){
                    dist[threadIdx.x] = new_dist;
                    keys[threadIdx.x] = i;
                }
            }
            __syncthreads();
        }

        #pragma unroll
        for (int i=0; i<k; ++i){
            array[i] = keys[i * (blockDim.x+1) + threadIdx.x];
        }

        #pragma unroll
        for (int i=0; i<k; ++i){
            keys[threadIdx.x * k + i] = array[i];
        }

        __syncthreads();

        if (k%2 == 0){
            #pragma unroll
            for (SizeT i=threadIdx.x; i<(blockDim.x*k)/2; i+=blockDim.x){
                reinterpret_cast<int2*>(keys_out + firstPoint*k)[i] = reinterpret_cast<int2*>(keys)[i];
            }
        }else{
            #pragma unroll
            for (SizeT i=threadIdx.x; i<blockDim.x*k; i+=blockDim.x){
                keys_out[firstPoint*k + i] = keys[i];
            }
        }
        __syncthreads();
    };

    auto knn_shared_transpose_op = 
    [num_points, k, dim, points, keys_out, data_size, points_size, dist_size, keys_size] 
    __device__ (ValueT* d, const SizeT &src, char* shared_mem){
                   
        ValueT* dist = (ValueT*) shared_mem;
        ValueT* b_sh_points = (ValueT*) (shared_mem + dist_size);
        int* keys = (int*) (shared_mem + dist_size + points_size);
        ValueT* sh_point = (ValueT*)(shared_mem + dist_size + points_size + keys_size);
        
        __shared__ int firstPoint;
        if (threadIdx.x == 0){
            firstPoint = src;
        }
        __syncthreads();

        ValueT* ptr = points + firstPoint;
        int idx = threadIdx.x;
        if (firstPoint + threadIdx.x < num_points){
            b_sh_points[idx] = ptr[threadIdx.x];
        }

        ptr += num_points;
        ValueT value = util::PreDefinedValues<ValueT>::InvalidValue;
        if (firstPoint + threadIdx.x < num_points){
            value = ptr[threadIdx.x];
        }

        // Initializations of basic points
        #pragma unroll
        for (SizeT i = 1; i < dim; ++i){
            ptr += num_points;
            int idx = i * (blockDim.x+1) + threadIdx.x;
            b_sh_points[idx] = value;
            value = util::PreDefinedValues<int>::InvalidValue;
            if (firstPoint + threadIdx.x < num_points){
            value = ptr[threadIdx.x];
            }
        }


        // Initializations of dist and keys
        #pragma unroll
        for (int i = 0; i < k; ++i){
            int idx = i * (blockDim.x+1) + threadIdx.x;
            dist[idx] = util::PreDefinedValues<ValueT>::MaxValue;
            keys[idx] = util::PreDefinedValues<int>::InvalidValue;
        }

        #pragma unroll
        for (SizeT i = 0; i<num_points; ++i){

            // Initialization of shared points (points [i...i*blocDim.x] in sh_points)
            // Proceeding points[[0..dim] * num_points + i];
            #pragma unroll 
            for (SizeT j=threadIdx.x; j<dim; j+=blockDim.x){
                // Doing better with fetching int4 data
                sh_point[j] = //b_sh_points[j];
                              points[j * num_points + i]; //from 500ms to 3500ms (transpose?)
                              //  points[i * dim + j]; //from 500ms to 3500ms (transpose?)
            }
            __syncthreads();
            
            ValueT new_dist = 0;
            if (src == i || src >= num_points) {
                new_dist = util::PreDefinedValues<ValueT>::MaxValue;
            } else {
                new_dist = euclidean_distance(dim, b_sh_points, (int)threadIdx.x, sh_point);
            } 

            //dist[threadIdx.x] = new_dist;
            // sorting 609ms to 5400ms 
            
            if (new_dist < dist[((k-1) * (blockDim.x+1)) + threadIdx.x]) {
                // new element is larger than the largest in distance array for "src" row
                // new_dist < dist[(k-1) * blockDim.x + threadIdx.x]
                SizeT current = k-1;
                #pragma unroll
                for (; current > 0; --current){
                    SizeT one_before = current-1;
                    if (new_dist >= dist[(one_before * (blockDim.x+1)) + threadIdx.x]){
                        dist[(current * (blockDim.x+1)) + threadIdx.x] = new_dist;
                        keys[(current * (blockDim.x+1)) + threadIdx.x] = i;
                        break;
                    } else {
                        dist[(current * (blockDim.x+1)) + threadIdx.x] = dist[(one_before * (blockDim.x+1)) + threadIdx.x];
                        keys[(current * (blockDim.x+1)) + threadIdx.x] = keys[(one_before * (blockDim.x+1)) + threadIdx.x];
                    }
                }
                if (current == (SizeT)0){
                    dist[threadIdx.x] = new_dist;
                    keys[threadIdx.x] = i;
                }
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i=0; i<blockDim.x; ++i){
            if (threadIdx.x < k){
                keys_out[(firstPoint + i) * k + threadIdx.x] = (ValueT)keys[threadIdx.x * (blockDim.x+1) + i];
            }
        }
        
        __syncthreads();
    };

    if (! USE_SHARED_MEM){
    
        debug("Used block size %d, grid size %d\n", block_size, grid_size);
        
        // Calculating theoretical occupancy
        int maxActiveBlocks;
        //cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, oprtr::SharedForAll_Kernel<decltype(distance_out), SizeT, decltype(knn_general_op)>, block_size, 0);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, oprtr::SharedForAll_Kernel<decltype(distance_out), SizeT, decltype(knn_half_op)>, block_size, (block_size * 12));
        debug("occupancy of SM is %d\n", maxActiveBlocks);

        // Checking rest of n-k points to choose k nearest.
        // Insertion n-k elements into sorted list
        // GUARD_CU(distance_out.SharedForAll(knn_general_op,
        GUARD_CU(distance_out.SharedForAll(knn_half_op,
            //num_points, target, stream, 64, 1024)); //time 82 min
            //num_points, target, stream, 128, 512)); //time 51.6 min
            //num_points, target, stream, 256, 256)); //time 44.12 min
            num_points, target, stream, block_size*(sizeof(ValueT)+sizeof(SizeT)), grid_size, block_size)); //time 41.32 min
            //num_points, target, stream, (block_size*8) + (block_size*4), grid_size, block_size)); //time 41.32 min
            //num_points, target, stream, shared_point_size, 512, 128)); //time 44.03 min

    }else{
        
        debug("Used threads %d, single data_size %d, shared memory %u, %d\n", block_size, data_size, shared_mem_size, sizeof(ValueT));
        debug("points size = %d, dist_size = %d, keys_size = %d, shared_point_size = %d\n", points_size, 
        dist_size, keys_size, shared_point_size);

        if (transpose){
            
            // Points is transposed
            //N M
            //   I1  I2  .. IN 
            //DA L1A L2A .. LNA
            //DB L1B L2B .. LNB
            //.. ..  ..  .. ..
            //DM L1M L2M .. LNM

            // Checking rest of n-k points to choose k nearest.
            // Insertion n-k elements into sorted list
            GUARD_CU2(distance_out.SharedForAll(knn_shared_transpose_op, num_points, target, stream, shared_mem_size, dim3(grid_size, 1, 1), dim3(block_size, 1, 1)), "shared for all failed");
        }else{
        
            // Points is not transposed
            //N M
            //   DA  DB  .. DM 
            //I1 L1A L1B .. L1M
            //I2 L2A L2B .. L2M
            //.. ..  ..  .. ..
            //IN LNA LNB .. LNM

            // Calculating theoretical occupancy
            int maxActiveBlocks;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, oprtr::SharedForAll_Kernel<decltype(distance_out), SizeT, decltype(knn_shared_not_transpose_op)>, block_size, shared_mem_size);
            debug("occupancy of SM is %d\n", maxActiveBlocks);

            // Checking rest of n-k points to choose k nearest.
            // Insertion n-k elements into sorted list
            GUARD_CU2(distance_out.SharedForAll(knn_shared_not_transpose_op, num_points, target, stream, shared_mem_size, grid_size, block_size), "shared for all failed");
        }
    }

    return retval;
  }

  /**
   * @brief Routine to combine received data and local data
   * @tparam NUM_VERTEX_ASSOCIATES Number of data associated with each
   * transmition item, typed VertexT
   * @tparam NUM_VALUE__ASSOCIATES Number of data associated with each
   * transmition item, typed ValueT
   * @param  received_length The numver of transmition items received
   * @param[in] peer_ which peer GPU the data came from
   * \return cudaError_t error message(s), if any
   */
  template <int NUM_VERTEX_ASSOCIATES, int NUM_VALUE__ASSOCIATES>
  cudaError_t ExpandIncoming(SizeT &received_length, int peer_) {
    // ================ INCOMPLETE TEMPLATE - MULTIGPU ====================

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // TODO: add problem specific data alias here, e.g.:
    // auto         &distance          =   data_slice.distance;

    auto expand_op = [
                         // TODO: pass data used by the lambda, e.g.:
                         // distance
    ] __host__ __device__(VertexT & key, const SizeT &in_pos,
                          VertexT *vertex_associate_ins,
                          ValueT *value__associate_ins) -> bool {
      // TODO: fill in the lambda to combine received and local data, e.g.:
      // ValueT in_val  = value__associate_ins[in_pos];
      // ValueT old_val = atomicMin(distance + key, in_val);
      // if (old_val <= in_val)
      //     return false;
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    auto it = this->enactor->enactor_slices[0].enactor_stats.iteration;
    if (it > 0)
      return true;
    else
      return false;
  }
};  // end of knnIteration

/**
 * @brief knn enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<
          typename _Problem::GraphT, typename _Problem::GraphT::VertexT,
          typename _Problem::GraphT::ValueT, ARRAY_FLAG, cudaHostRegisterFlag> {
 public:
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::GraphT GraphT;
  typedef typename GraphT::VertexT LabelT;
  typedef typename GraphT::ValueT ValueT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef knnIterationLoop<EnactorT> IterationT;

  Problem *problem;
  IterationT *iterations;

  /**
   * @brief knn constructor
   */
  Enactor() : BaseEnactor("KNN"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief knn destructor
   */
  virtual ~Enactor() { /*Release();*/
  }

  /*
   * @brief Releasing allocated memory space
   * @param target The location to release memory from
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Release(target));
    delete[] iterations;
    iterations = NULL;
    problem = NULL;
    return retval;
  }

  /**
   * @brief Initialize the problem.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    // Lazy initialization
    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 2, NULL,target, false));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(1, 1, this->queue_factors));
    }

    iterations = new IterationT[this->num_gpus];
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(iterations[gpu].Init(this, gpu));
    }

    GUARD_CU(this->Init_Threads(
        this, (CUT_THREADROUTINE) & (GunrockThread<EnactorT>)));
    return retval;
  }

  /**
   * @brief one run of knn, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<
        // change to how many {VertexT, ValueT} data need to communicate
        //       per element in the inter-GPU sub-frontiers
        0, 1, IterationT>(thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Reset enactor
...
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(SizeT n, SizeT k, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    typedef typename GraphT::VertexT VertexT;
    cudaError_t retval = cudaSuccess;

    GUARD_CU(BaseEnactor::Reset(target));

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if (this->num_gpus == 1) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          if (peer_ == 0) {
            GUARD_CU(frontier.V_Q()->ForEach(
                [] __host__ __device__(VertexT & v) { v = 0; },
                1, target, 0));
          }
        }
      } else {
        // MULTIGPU INCOMPLETE
      }
    }

    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief Enacts a knn computing on the specified graph.
...
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU KNN Done.", this->flag & Debug);
    return retval;
  }
};

}  // namespace knn
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
