// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sage_enactor.cuh
 *
 * @brief SSSP Problem Enactor
 */

#pragma once

#include <gunrock/app/enactor_base.cuh>
#include <gunrock/app/enactor_iteration.cuh>
#include <gunrock/app/enactor_loop.cuh>
#include <gunrock/app/sage/sage_problem.cuh>
#include <gunrock/oprtr/oprtr.cuh>

namespace gunrock {
namespace app {
namespace sage {

/**
 * @brief Speciflying parameters for SSSP Enactor
 * @param parameters The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_enactor(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;
  GUARD_CU(app::UseParameters_enactor(parameters));
  return retval;
}

static const cub::CacheLoadModifier W_LOAD = cub::LOAD_LDG;    // for Wa and Wf
static const cub::CacheLoadModifier F_LOAD = cub::LOAD_LDG;    // for features
static const cub::CacheLoadModifier S_LOAD = cub::LOAD_CA;     // for Sums
static const cub::CacheStoreModifier S_STORE = cub::STORE_WB;  // for Sums
static const cub::CacheLoadModifier T_LOAD = cub::LOAD_CA;     // for temps
static const cub::CacheStoreModifier T_STORE = cub::STORE_WB;  // for temps

template <typename GraphT, typename ValueT>
__global__ void sage_kernel1(typename GraphT::VertexT source_start,
                             int num_children_per_source, const GraphT graph,
                             uint64_t feature_column, ValueT *features,
                             int num_leafs_per_child, ValueT *sums,
                             curandState *rand_states, ValueT *sums_child_feat,
                             typename GraphT::VertexT *children,
                             typename GraphT::SizeT num_children) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT child_num = blockIdx.x;
  extern __shared__ VertexT s_leafs[];
  __shared__ VertexT s_child;
  __shared__ SizeT s_child_degree, s_child_edge_offset;
  SizeT thread_id = (SizeT)blockIdx.x * blockDim.x + threadIdx.x;

  while (child_num < num_children) {
    if (threadIdx.x == 0) {
      VertexT source = child_num / num_children_per_source + source_start;
      s_child = graph.GetEdgeDest(graph.GetNeighborListOffset(source) +
                                  curand_uniform(rand_states + thread_id) *
                                      graph.GetNeighborListLength(source));
      children[child_num] = s_child;
      s_child_degree = graph.GetNeighborListLength(s_child);
      s_child_edge_offset = graph.GetNeighborListOffset(s_child);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < num_leafs_per_child; i += blockDim.x) {
      s_leafs[i] = graph.GetEdgeDest(s_child_edge_offset +
                                     curand_uniform(rand_states + thread_id) *
                                         s_child_degree);
    }
    __syncthreads();

    for (auto i = threadIdx.x; i < feature_column; i += blockDim.x) {
      ValueT sum = 0;
      for (int j = 0; j < num_leafs_per_child; j++)
        sum += Load<F_LOAD>(features + (s_leafs[j] * feature_column + i));
      sum /= num_leafs_per_child;
      Store<S_STORE>(sums + (child_num * feature_column + i), sum);

      atomicAdd(sums_child_feat +
                    (child_num / num_children_per_source * feature_column + i),
                Load<F_LOAD>(features + (s_child * feature_column + i)) /
                    num_children_per_source);
    }
    __syncthreads();
    child_num += gridDim.x;
  }
}

template <int LOG_THREADS_, typename VertexT, typename SizeT, typename ValueT>
__global__ void sage_kernel2(int num_children_per_source,
                             uint64_t feature_column, ValueT *features,
                             ValueT *W_f_1, int Wf1_dim1, VertexT *children,
                             ValueT *W_a_1, int Wa1_dim1, int Wa2_dim0,
                             int Wf2_dim0, ValueT *children_temp,
                             ValueT *sums_child_feat, ValueT *sums,
                             SizeT num_children) {
  typedef util::reduce::BlockReduce<ValueT, LOG_THREADS_> BlockReduceT;
  __shared__ VertexT s_child;
  __shared__ typename BlockReduceT::TempSpace reduce_space;
  SizeT child_num = blockIdx.x;

  while (child_num < num_children) {
    if (threadIdx.x == 0) {
      s_child = children[child_num];
    }
    __syncthreads();

    ValueT val = 0;
    if (threadIdx.x < Wf1_dim1) {
      auto f_offset = s_child * feature_column;
      for (int f = 0; f < feature_column; f++)
        val += Load<F_LOAD>(features + f_offset + f) *
               Load<W_LOAD>(W_f_1 + (f * Wf1_dim1 + threadIdx.x));
    } else if (threadIdx.x < Wf1_dim1 + Wa1_dim1) {
      auto f_offset = child_num * feature_column;
      for (int f = 0; f < feature_column; f++)
        val += Load<cub::LOAD_LDG>(sums + f_offset + f) *
               Load<W_LOAD>(W_a_1 + (f * Wa1_dim1 + threadIdx.x - Wf1_dim1));
    }
    if (val < 0) val = 0;  // relu()
    double L2_child_temp = BlockReduceT::Reduce(
        val * val, [](const ValueT &a, const ValueT &b) { return a + b; },
        (ValueT)0, reduce_space);
    if (threadIdx.x < Wa2_dim0) {
      L2_child_temp = 1.0 / sqrt(L2_child_temp);
      val *= L2_child_temp;
      atomicAdd(children_temp +
                    (child_num / num_children_per_source) * Wa2_dim0 +
                    threadIdx.x,
                val / num_children_per_source);
    }

    __syncthreads();
    child_num += gridDim.x;
  }
}

template <int LOG_THREADS_, typename SizeT, typename VertexT, typename ValueT>
__global__ void sage_kernel3(uint64_t feature_column, ValueT *features,
                             VertexT source_start, ValueT *W_f_1, int Wf1_dim1,
                             ValueT *children_temp, ValueT *sums_child_feat,
                             ValueT *W_a_1, int Wa1_dim1, ValueT *W_f_2,
                             int Wf2_dim1, int Wf2_dim0, ValueT *W_a_2,
                             int Wa2_dim1, int Wa2_dim0, ValueT *source_result,
                             int result_column, ValueT *source_temp,
                             VertexT num_sources, bool use_shared_source_temp) {
  typedef util::reduce::BlockReduce<ValueT, LOG_THREADS_> BlockReduceT;
  __shared__ typename BlockReduceT::TempSpace reduce_space;
  __shared__ double s_L2;
  extern __shared__ ValueT s_source_temp[];
  VertexT source_num = blockIdx.x;

  while (source_num < num_sources) {
    ValueT val = 0;

    if (threadIdx.x < Wf1_dim1) {
      auto f_offset = (source_start + source_num) * feature_column;
      for (int f = 0; f < feature_column; f++)
        val += Load<F_LOAD>(features + f_offset + f) *
               Load<W_LOAD>(W_f_1 + (f * Wf1_dim1 + threadIdx.x));
    } else if (threadIdx.x < Wf2_dim0) {
      auto f_offset = source_num * feature_column;
      for (int f = 0; f < feature_column; f++)
        val += Load<S_LOAD>(sums_child_feat + f_offset + f) *
               Load<W_LOAD>(W_a_1 + (f * Wa1_dim1 + threadIdx.x - Wf1_dim1));
    }
    if (val < 0) val = 0;  // relu()
    double L2 = BlockReduceT::Reduce(
        val * val, [](const ValueT &a, const ValueT &b) { return a + b; },
        (ValueT)0, reduce_space);
    if (threadIdx.x == 0) s_L2 = 1.0 / sqrt(L2);
    __syncthreads();

    if (threadIdx.x < Wf2_dim0) {
      // L2 = 1.0 / sqrt(L2);
      if (use_shared_source_temp)
        s_source_temp[threadIdx.x] = val * s_L2;
      else
        Store<T_STORE>(source_temp + (source_num * Wf2_dim0 + threadIdx.x),
                       (ValueT)(val * s_L2));
    }
    __syncthreads();

    val = 0;
    if (threadIdx.x < Wf2_dim1) {
      SizeT offset = source_num * Wf2_dim0;
      for (int y = 0; y < Wf2_dim0; y++)
        val +=
            (use_shared_source_temp ? s_source_temp[y]
                                    : Load<T_LOAD>(source_temp + offset + y)) *
            Load<W_LOAD>(W_f_2 + (y * Wf2_dim1 + threadIdx.x));
    } else if (threadIdx.x < result_column) {
      SizeT offset = source_num * Wa2_dim0;
      for (int y = 0; y < Wa2_dim0; y++)
        val += Load<cub::LOAD_LDG>(children_temp + offset + y) *
               Load<W_LOAD>(W_a_2 + (y * Wa2_dim1 + threadIdx.x - Wf2_dim1));
    }
    if (val < 0) val = 0;
    L2 = BlockReduceT::Reduce(
        val * val, [](const ValueT &a, const ValueT &b) { return a + b; },
        (ValueT)0, reduce_space);
    if (threadIdx.x == 0) s_L2 = 1.0 / sqrt(L2);
    __syncthreads();
    if (threadIdx.x < result_column) {
      // L2 = 1.0 / sqrt(L2);
      Store<cub::STORE_WT>(
          source_result + (source_num * result_column + threadIdx.x),
          (ValueT)(val * s_L2));
    }
    __syncthreads();
    source_num += gridDim.x;
  }
}

/**
 * @brief defination of SAGE iteration loop
 * @tparam EnactorT Type of enactor
 */
template <typename EnactorT>
struct SAGEIterationLoop
    : public IterationLoopBase<EnactorT, Use_FullQ | Push
                               // |
                               // (((EnactorT::Problem::FLAG &
                               // Mark_Predecessors) != 0) ?
                               //  Update_Predecessors : 0x0)
                               > {
  typedef typename EnactorT::VertexT VertexT;
  typedef typename EnactorT::SizeT SizeT;
  typedef typename EnactorT::ValueT ValueT;
  typedef typename EnactorT::Problem::GraphT::CooT CooT;
  typedef typename EnactorT::Problem::GraphT::GpT GpT;
  typedef IterationLoopBase<EnactorT, Use_FullQ | Push
                            // |
                            // (((EnactorT::Problem::FLAG & Mark_Predecessors)
                            // != 0) ?
                            //  Update_Predecessors : 0x0)
                            >
      BaseIterationLoop;

  SAGEIterationLoop() : BaseIterationLoop() {}

  /**
   * @brief Core computation of sage, one iteration
   * @param[in] peer_ Which GPU peers to work on, 0 means local
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Core(int peer_ = 0) {
    // Data sage that works on
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    auto &enactor_stats = enactor_slice.enactor_stats;
    auto &graph = data_slice.sub_graph[0];
    auto &W_f_1 = data_slice.W_f_1_1D;
    auto Wf1_dim1 = data_slice.Wf1_dim1;
    auto &W_a_1 = data_slice.W_a_1_1D;
    auto Wa1_dim1 = data_slice.Wa1_dim1;
    auto &W_f_2 = data_slice.W_f_2_1D;
    auto Wf2_dim0 = data_slice.Wf2_dim0;
    auto Wf2_dim1 = data_slice.Wf2_dim1;
    auto &W_a_2 = data_slice.W_a_2_1D;
    auto Wa2_dim0 = data_slice.Wa2_dim0;
    auto Wa2_dim1 = data_slice.Wa2_dim1;
    auto &features = data_slice.features_1D;
    uint64_t feature_column = data_slice.feature_column;
    auto &source_result = data_slice.source_result;
    auto result_column = data_slice.result_column;
    auto num_children_per_source = data_slice.num_children_per_source;
    auto num_leafs_per_child = data_slice.num_leafs_per_child;
    auto &sums = data_slice.sums;
    auto &sums_child_feat = data_slice.sums_child_feat;
    // auto         &child_temp         =   data_slice.child_temp;
    auto &children_temp = data_slice.children_temp;
    auto &children = data_slice.children;
    auto &rand_states = data_slice.rand_states;
    auto &retval = enactor_stats.retval;
    auto &stream = enactor_slice.stream;
    auto &iteration = enactor_stats.iteration;
    VertexT source_start = iteration * data_slice.batch_size;
    VertexT source_end = (iteration + 1) * data_slice.batch_size;
    if (source_end >= graph.nodes) source_end = graph.nodes;
    VertexT num_sources = source_end - source_start;
    SizeT num_children = num_sources * data_slice.num_children_per_source;

    util::PrintMsg("Processing sources [" + std::to_string(source_start) +
                       ", " + std::to_string(source_start + num_sources) + ")",
                   data_slice.debug);
    GUARD_CU(
        children_temp.ForEach([] __host__ __device__(ValueT & val) { val = 0; },
                              num_sources * Wf2_dim0, util::DEVICE, stream));

    GUARD_CU(sums_child_feat.ForEach(
        [] __host__ __device__(ValueT & val) { val = 0; },
        num_sources * feature_column, util::DEVICE, stream));

    if (data_slice.custom_kernels) {
      sage_kernel1<<<2560, min((int)feature_column, 512),
                     num_leafs_per_child * sizeof(VertexT), stream>>>(
          source_start, num_children_per_source, graph, feature_column,
          features.GetPointer(util::DEVICE), num_leafs_per_child,
          sums.GetPointer(util::DEVICE), rand_states.GetPointer(util::DEVICE),
          sums_child_feat.GetPointer(util::DEVICE),
          children.GetPointer(util::DEVICE), num_children);
    } else {
      int grid_size = 80;
      int block_size = 256;
      GUARD_CU(children.ForAll(
          [source_start, num_children_per_source, graph, feature_column,
           features, num_leafs_per_child, sums, rand_states, sums_child_feat,
           grid_size,
           block_size] __host__ __device__(VertexT * childs, const SizeT &i) {
            VertexT source = i / num_children_per_source + source_start;
            // SizeT   offset = curand_uniform(rand_states + i)
            //    * graph.GetNeighborListLength(source);
            // SizeT   edge   = graph.GetNeighborListOffset(source) + offset;
            // VertexT child  = graph.GetEdgeDest(edge);
            VertexT child = graph.GetEdgeDest(
                graph.GetNeighborListOffset(source) +
                curand_uniform(rand_states + (i % (grid_size * block_size))) *
                    graph.GetNeighborListLength(source));
            childs[i] = child;
            SizeT child_degree = graph.GetNeighborListLength(child);
            SizeT child_edge_offset = graph.GetNeighborListOffset(child);
            // float sums [64] = {0.0} ; //local vector

            auto f_offset = i * feature_column;
            for (auto f = 0; f < feature_column; f++)
              Store<S_STORE>(sums + (f_offset + f), (ValueT)0);
            for (int j = 0; j < num_leafs_per_child; j++) {
              // SizeT   offset2 = 0;//cuRand() * child_degree;
              // SizeT   edge2   = graph.GetNeighborListOffset(child)
              //    + curand_uniform(rand_states + i) * child_degree;
              // VertexT leaf    = graph.GetEdgeDest(edge2);
              VertexT leaf = graph.GetEdgeDest(
                  child_edge_offset +
                  curand_uniform(rand_states + (i % (grid_size * block_size))) *
                      child_degree);
              auto offset = leaf * feature_column;

              for (auto f = 0; f < feature_column; f++) {
                Store<S_STORE>(sums + (f_offset + f),
                               Load<S_LOAD>(sums + (f_offset + f)) +
                                   Load<F_LOAD>(features + (offset + f)));
                /// num_neigh2;// merged line 176 171
              }
            }
            for (auto f = 0; f < feature_column; f++)
              Store<S_STORE>(
                  sums + (f_offset + f),
                  Load<S_LOAD>(sums + (f_offset + f)) / num_leafs_per_child);
            // agg feaures for leaf nodes alg2 line 11 k = 1;

            auto offset = i / num_children_per_source * feature_column;
            f_offset = child * feature_column;
            // SizeT f_offset = children[i] * feature_column;
            for (auto f = 0; f < feature_column; f++) {
              atomicAdd(sums_child_feat + offset + f,
                        Load<F_LOAD>(features + (f_offset + f)) /
                            num_children_per_source);
              // merge 220 and 226
            }
          },
          num_children, util::DEVICE, stream, 80, 256));
    }
    // GUARD_CU2(cudaDeviceSynchronize(),
    //    "cudaDeviceSynchronize failed.");

    if (data_slice.custom_kernels && Wa2_dim0 <= 1024) {
      if (Wa2_dim0 <= 128)
        sage_kernel2<7><<<1280, 128, 0, stream>>>(
            num_children_per_source, feature_column,
            features.GetPointer(util::DEVICE), W_f_1.GetPointer(util::DEVICE),
            Wf1_dim1, children.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1, Wa2_dim0, Wf2_dim0,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            sums.GetPointer(util::DEVICE), num_children);

      else if (Wa2_dim0 <= 256)
        sage_kernel2<8><<<1280, 256, 0, stream>>>(
            num_children_per_source, feature_column,
            features.GetPointer(util::DEVICE), W_f_1.GetPointer(util::DEVICE),
            Wf1_dim1, children.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1, Wa2_dim0, Wf2_dim0,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            sums.GetPointer(util::DEVICE), num_children);

      else if (Wa2_dim0 <= 512)
        sage_kernel2<9><<<1280, 512, 0, stream>>>(
            num_children_per_source, feature_column,
            features.GetPointer(util::DEVICE), W_f_1.GetPointer(util::DEVICE),
            Wf1_dim1, children.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1, Wa2_dim0, Wf2_dim0,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            sums.GetPointer(util::DEVICE), num_children);

      else if (Wa2_dim0 <= 1024)
        sage_kernel2<10><<<1280, 1024, 0, stream>>>(
            num_children_per_source, feature_column,
            features.GetPointer(util::DEVICE), W_f_1.GetPointer(util::DEVICE),
            Wf1_dim1, children.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1, Wa2_dim0, Wf2_dim0,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            sums.GetPointer(util::DEVICE), num_children);

    } else {
      GUARD_CU(data_slice.child_temp.ForAll(
          [num_children_per_source, feature_column, features, W_f_1, Wf1_dim1,
           children, W_a_1, Wa1_dim1, Wa2_dim0, Wf2_dim0, children_temp,
           sums_child_feat,
           sums] __host__ __device__(ValueT * child_temp_, const SizeT &i) {
            ValueT *child_temp = child_temp_ + i * Wf2_dim0;
            auto f_offset = children[i] * feature_column;
            double L2_child_temp = 0.0;
            for (int x = 0; x < Wf1_dim1; x++) {
              ValueT val = 0;
              for (auto f = 0; f < feature_column; f++)
                val += Load<F_LOAD>(features + (f_offset + f)) *
                       Load<W_LOAD>(W_f_1 + (f * Wf1_dim1 + x));
              if (val < 0)  // relu()
                val = 0;
              L2_child_temp += val * val;
              Store<T_STORE>(child_temp + x, val);
            }  // got 1st half of h_B1^1

            auto offset = i * feature_column;
            for (int x = 0; x < Wa1_dim1; x++) {
              ValueT val = 0;
              for (auto f = 0; f < feature_column; f++)
                val += Load<cub::LOAD_LDG>(sums + (offset + f)) *
                       Load<W_LOAD>(W_a_1 + (f * Wa1_dim1 + x));
              if (val < 0)  // relu()
                val = 0;
              L2_child_temp += val * val;
              Store<T_STORE>(child_temp + (x + Wf1_dim1), val);
            }  // got 2nd half of h_B1^1

            // activation and L-2 normalize
            // double L2_child_temp = 0.0;
            // for (int x =0; x < Wa2_dim0; x++)
            //{
            //    ValueT val = child_temp[x];
            //    if (val < 0) // relu()
            //        val = 0;
            //    L2_child_temp += val * val;
            //    child_temp[x] = val;
            //}  //finished relu
            L2_child_temp = 1.0 / sqrt(L2_child_temp);
            offset = i / num_children_per_source * Wa2_dim0;
            for (int x = 0; x < Wa2_dim0; x++) {
              // child_temp[idx_0] = child_temp[idx_0] /sqrt (L2_child_temp);
              // child_temp[x] *= L2_child_temp;
              ValueT val = Load<T_LOAD>(child_temp + x);
              val *= L2_child_temp;
              //}//finished L-2 norm, got h_B1^1, algo2 line13

              // add the h_B1^1 to children_temp, also agg it
              // for (int x =0; x < Wa2_dim0; x ++ ) //205
              //{
              atomicAdd(children_temp + (offset + x),
                        val / num_children_per_source);
            }  // finished agg (h_B1^1)

            // end of for each child
          },
          num_children, util::DEVICE, stream, 80));
    }
    // GUARD_CU2(cudaDeviceSynchronize(),
    //    "cudaDeviceSynchronize failed.");

    if (iteration != 0) {
      GUARD_CU2(cudaStreamWaitEvent(stream, data_slice.d2h_finish, 0),
                "cudaStreamWaitEvent failed");
    }
    int max_dim = max(Wf1_dim1 + Wa1_dim1, Wf2_dim1 + Wa2_dim1);
    if (data_slice.custom_kernels && max_dim <= 1024) {
      size_t shared_size = Wf2_dim0 * sizeof(ValueT);
      bool use_shared_source_temp = (shared_size <= 24 * 1024);
      if (!use_shared_source_temp) shared_size = 0;
      if (max_dim <= 128)
        sage_kernel3<7, SizeT><<<1280, 128, shared_size, stream>>>(
            feature_column, features.GetPointer(util::DEVICE), source_start,
            W_f_1.GetPointer(util::DEVICE), Wf1_dim1,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1,
            W_f_2.GetPointer(util::DEVICE), Wf2_dim1, Wf2_dim0,
            W_a_2.GetPointer(util::DEVICE), Wa2_dim1, Wa2_dim0,
            source_result.GetPointer(util::DEVICE), result_column,
            data_slice.source_temp.GetPointer(util::DEVICE), num_sources,
            use_shared_source_temp);

      else if (max_dim <= 256)
        sage_kernel3<8, SizeT><<<1280, 256, shared_size, stream>>>(
            feature_column, features.GetPointer(util::DEVICE), source_start,
            W_f_1.GetPointer(util::DEVICE), Wf1_dim1,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1,
            W_f_2.GetPointer(util::DEVICE), Wf2_dim1, Wf2_dim0,
            W_a_2.GetPointer(util::DEVICE), Wa2_dim1, Wa2_dim0,
            source_result.GetPointer(util::DEVICE), result_column,
            data_slice.source_temp.GetPointer(util::DEVICE), num_sources,
            use_shared_source_temp);

      else if (max_dim <= 512)
        sage_kernel3<9, SizeT><<<1280, 512, shared_size, stream>>>(
            feature_column, features.GetPointer(util::DEVICE), source_start,
            W_f_1.GetPointer(util::DEVICE), Wf1_dim1,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1,
            W_f_2.GetPointer(util::DEVICE), Wf2_dim1, Wf2_dim0,
            W_a_2.GetPointer(util::DEVICE), Wa2_dim1, Wa2_dim0,
            source_result.GetPointer(util::DEVICE), result_column,
            data_slice.source_temp.GetPointer(util::DEVICE), num_sources,
            use_shared_source_temp);

      else if (max_dim <= 1024)
        sage_kernel3<10, SizeT><<<1280, 1024, shared_size, stream>>>(
            feature_column, features.GetPointer(util::DEVICE), source_start,
            W_f_1.GetPointer(util::DEVICE), Wf1_dim1,
            children_temp.GetPointer(util::DEVICE),
            sums_child_feat.GetPointer(util::DEVICE),
            W_a_1.GetPointer(util::DEVICE), Wa1_dim1,
            W_f_2.GetPointer(util::DEVICE), Wf2_dim1, Wf2_dim0,
            W_a_2.GetPointer(util::DEVICE), Wa2_dim1, Wa2_dim0,
            source_result.GetPointer(util::DEVICE), result_column,
            data_slice.source_temp.GetPointer(util::DEVICE), num_sources,
            use_shared_source_temp);
    } else {
      GUARD_CU(data_slice.source_temp.ForAll(
          [feature_column, features, source_start, W_f_1, Wf1_dim1,
           children_temp, sums_child_feat, W_a_1, Wa1_dim1, W_f_2, Wf2_dim1,
           Wf2_dim0, W_a_2, Wa2_dim1, Wa2_dim0, source_result,
           result_column] __host__ __device__(ValueT * source_temp_,
                                              const SizeT &i) {
            ValueT *source_temp = source_temp_ + i * Wf2_dim0;
            VertexT source = source_start + i;
            auto offset = source * feature_column;
            // get ebedding vector for child node (h_{B2}^{1}) alg2 line 12
            double L2_source_temp = 0.0;
            for (int x = 0; x < Wf1_dim1; x++) {
              ValueT val = 0;
              for (auto f = 0; f < feature_column; f++)
                val += Load<F_LOAD>(features + (offset + f)) *
                       Load<W_LOAD>(W_f_1 + (f * Wf1_dim1 + x));
              if (val < 0) val = 0;  // relu()
              L2_source_temp += val * val;
              Store<T_STORE>(source_temp + x, val);
            }  // got 1st half of h_B2^1

            offset = i * feature_column;
            for (int x = 0; x < Wa1_dim1; x++) {
              ValueT val = 0;
              for (auto f = 0; f < feature_column; f++)
                val += sums_child_feat[offset + f] *
                       Load<W_LOAD>(W_a_1 + (f * Wa1_dim1 + x));
              if (val < 0) val = 0;  // relu()
              L2_source_temp += val * val;
              Store<T_STORE>(source_temp + (Wf1_dim1 + x), val);
            }  // got 2nd half of h_B2^1

            // for (int x =0; x < Wf2_dim0; x++)
            //{
            //    ValueT val = source_temp[x];
            //    if (val < 0)
            //        val = 0; // relu()
            //    L2_source_temp += val * val;
            //    source_temp[x] = val;
            //} //finished relu
            L2_source_temp = 1.0 / sqrt(L2_source_temp);
            for (int x = 0; x < Wf2_dim0; x++) {
              // source_temp[idx_0] = source_temp[idx_0] /sqrt (L2_source_temp);
              // source_temp[x] *= L2_source_temp;
              Store<T_STORE>(
                  source_temp + x,
                  (ValueT)(Load<T_LOAD>(source_temp + x) * L2_source_temp));
            }  // finished L-2 norm for source temp

            //////////////////////////////////////////////////////////////////////////////////////
            // get h_B2^2 k =2.
            offset = i * result_column;
            double L2_source_result = 0.0;
            for (int x = 0; x < Wf2_dim1; x++) {
              ValueT val = 0;  // source_result[offset + x];
              // printf ("source_r1_0:%f", source_result[idx_0] );
              for (int y = 0; y < Wf2_dim0; y++)
                val += Load<T_LOAD>(source_temp + y)  // source_temp[y]
                       * Load<W_LOAD>(W_f_2 + (y * Wf2_dim1 + x));
              if (val < 0) val = 0;  // relu()
              L2_source_result += val * val;
              Store<T_STORE>(source_result + (offset + x), val);
              // printf ("source_r1:%f", source_result[idx_0] );
            }  // got 1st half of h_B2^2

            for (int x = 0; x < Wa2_dim1; x++) {
              // printf ("source_r2_0:%f", source_result[idx_0] );
              ValueT val = 0;  // source_result[offset + x];
              for (int y = 0; y < Wa2_dim0; y++)
                val += Load<cub::LOAD_LDG>(children_temp + i * Wa2_dim0 + y)
                       // children_temp[i * Wa2_dim0 + y]
                       * Load<W_LOAD>(W_a_2 + (y * Wa2_dim1 + x));
              if (val < 0) val = 0;  // relu()
              L2_source_result += val * val;
              Store<T_STORE>(source_result + (offset + Wf2_dim1 + x), val);
            }  // got 2nd half of h_B2^2

            // for (int x =0; x < result_column; x ++ )
            //{
            //    ValueT val = source_result[offset + x];
            //    if (val < 0) // relu()
            //        val = 0;
            //    L2_source_result += val * val;
            //    source_result[offset + x] = val;
            //} //finished relu
            L2_source_result = 1.0 / sqrt(L2_source_result);
            for (int x = 0; x < result_column; x++) {
              // source_result[offset + x] *= L2_source_result;
              Store<cub::STORE_WT>(
                  source_result + (offset + x),
                  (ValueT)(Load<T_LOAD>(source_result + (offset + x)) *
                           L2_source_result));
              // printf ("source_r:%f", source_result[idx_0] );
              // printf ("ch_t:%f", children_temp[idx_0]);
            }  // finished L-2 norm for source result
          },
          num_sources, util::DEVICE, stream, 640));
    }

    // GUARD_CU2(cudaDeviceSynchronize(),
    //    "cudaDeviceSynchronize failed.");
    GUARD_CU2(cudaEventRecord(data_slice.d2h_start, stream),
              "cudaEventRecord failed.");
    GUARD_CU2(
        cudaStreamWaitEvent(data_slice.d2h_stream, data_slice.d2h_start, 0),
        "cudaStreamWaitEvent failed.");
    GUARD_CU2(cudaMemcpyAsync(
                  data_slice.host_source_result +
                      (((uint64_t)source_start) * result_column),
                  source_result.GetPointer(util::DEVICE),
                  ((uint64_t)num_sources) * result_column * sizeof(ValueT),
                  cudaMemcpyDeviceToHost, data_slice.d2h_stream),
              "source_result D2H copy failed");
    GUARD_CU2(cudaEventRecord(data_slice.d2h_finish, data_slice.d2h_stream),
              "cudaEventRecord failed.");
    // GUARD_CU2(cudaDeviceSynchronize(),
    //    "cudaDeviceSynchronize failed.");

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
    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor
            ->enactor_slices[this->gpu_num * this->enactor->num_gpus + peer_];
    // auto iteration = enactor_slice.enactor_stats.iteration;
    // auto         &distances          =   data_slice.distances;
    // auto         &labels             =   data_slice.labels;
    // auto         &preds              =   data_slice.preds;
    // auto          label              =   this -> enactor ->
    //    mgpu_slices[this -> gpu_num].in_iteration[iteration % 2][peer_];

    auto expand_op = [] __host__ __device__(
                         VertexT & key, const SizeT &in_pos,
                         VertexT *vertex_associate_ins,
                         ValueT *value__associate_ins) -> bool {
      /*
      ValueT in_val  = value__associate_ins[in_pos];
      ValueT old_val = atomicMin(distances + key, in_val);
      if (old_val <= in_val)
          return false;
      if (labels[key] == label)
          return false;
      labels[key] = label;
      if (!preds.isEmpty())
          preds[key] = vertex_associate_ins[in_pos];
      */
      return true;
    };

    cudaError_t retval =
        BaseIterationLoop::template ExpandIncomingBase<NUM_VERTEX_ASSOCIATES,
                                                       NUM_VALUE__ASSOCIATES>(
            received_length, peer_, expand_op);
    return retval;
  }

  bool Stop_Condition(int gpu_num = 0) {
    int num_gpus = this->enactor->num_gpus;
    auto &enactor_slices = this->enactor->enactor_slices;

    for (int gpu = 0; gpu < num_gpus * num_gpus; gpu++) {
      auto &retval = enactor_slices[gpu].enactor_stats.retval;
      if (retval == cudaSuccess) continue;
      printf("(CUDA error %d @ GPU %d: %s\n", retval, gpu % num_gpus,
             cudaGetErrorString(retval));
      fflush(stdout);
      return true;
    }

    auto &data_slice = this->enactor->problem->data_slices[this->gpu_num][0];
    auto &enactor_slice =
        this->enactor->enactor_slices[this->gpu_num * this->enactor->num_gpus];
    // util::PrintMsg("iter = " +
    // std::to_string(enactor_slice.enactor_stats.iteration)
    //    + ", batch_size = " + std::to_string(data_slice.batch_size)
    //    + ", nodes = " + std::to_string(data_slice.sub_graph -> nodes));
    if (enactor_slice.enactor_stats.iteration * data_slice.batch_size <
        data_slice.sub_graph->nodes)
      return false;
    return true;
  }

  cudaError_t Compute_OutputLength(int peer_) { return cudaSuccess; }

  cudaError_t Check_Queue_Size(int peer_) { return cudaSuccess; }
};  // end of SSSPIteration

/**
 * @brief SSSP enactor class.
 * @tparam _Problem Problem type we process on
 * @tparam ARRAY_FLAG Flags for util::Array1D used in the enactor
 * @tparam cudaHostRegisterFlag Flags for util::Array1D used in the enactor
 */
template <typename _Problem, util::ArrayFlag ARRAY_FLAG = util::ARRAY_NONE,
          unsigned int cudaHostRegisterFlag = cudaHostRegisterDefault>
class Enactor
    : public EnactorBase<typename _Problem::GraphT, typename _Problem::VertexT,
                         typename _Problem::ValueT, ARRAY_FLAG,
                         cudaHostRegisterFlag> {
 public:
  // Definations
  typedef _Problem Problem;
  typedef typename Problem::SizeT SizeT;
  typedef typename Problem::VertexT VertexT;
  typedef typename Problem::ValueT ValueT;
  typedef typename Problem::GraphT GraphT;
  typedef typename Problem::LabelT LabelT;
  typedef EnactorBase<GraphT, LabelT, ValueT, ARRAY_FLAG, cudaHostRegisterFlag>
      BaseEnactor;
  typedef Enactor<Problem, ARRAY_FLAG, cudaHostRegisterFlag> EnactorT;
  typedef SAGEIterationLoop<EnactorT> IterationT;

  // Members
  Problem *problem;
  IterationT *iterations;

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief SSSPEnactor constructor
   */
  Enactor() : BaseEnactor("sage"), problem(NULL) {
    this->max_num_vertex_associates = 0;
    this->max_num_value__associates = 1;
  }

  /**
   * @brief SSSPEnactor destructor
   */
  virtual ~Enactor() {
    // Release();
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
   * @brief Initialize the enactor.
   * @param[in] problem The problem object.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Init(Problem &problem, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    this->problem = &problem;

    GUARD_CU(BaseEnactor::Init(problem, Enactor_None, 0, NULL, target, false));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      auto &enactor_slice = this->enactor_slices[gpu * this->num_gpus + 0];
      auto &graph = problem.sub_graphs[gpu];
      GUARD_CU(enactor_slice.frontier.Allocate(graph.nodes, graph.edges,
                                               this->queue_factors));
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
   * @brief Reset enactor
   * @param[in] src Source node to start primitive.
   * @param[in] target Target location of data
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    typedef typename GraphT::GpT GpT;
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseEnactor::Reset(target));
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      if ((this->num_gpus == 1) ||
          (gpu == this->problem->org_graph->GpT::partition_table[src])) {
        this->thread_slices[gpu].init_size = 1;
        for (int peer_ = 0; peer_ < this->num_gpus; peer_++) {
          auto &frontier =
              this->enactor_slices[gpu * this->num_gpus + peer_].frontier;
          frontier.queue_length = (peer_ == 0) ? 1 : 0;
          // if (peer_ == 0)
          //{
          //    GUARD_CU(frontier.V_Q() -> ForEach(
          //        [src]__host__ __device__ (VertexT &v)
          //
          //        v = src;
          //    }
          //}
        }
      }

      // else {
      //    this -> thread_slices[gpu].init_size = 0;
      //    for (int peer_ = 0; peer_ < this -> num_gpus; peer_++)
      //    {
      //        this -> enactor_slices[gpu * this -> num_gpus + peer_]
      //            .frontier.queue_length = 0;
      //    }
      // }
    }
    GUARD_CU(BaseEnactor::Sync());
    return retval;
  }

  /**
   * @brief one run of sage, to be called within GunrockThread
   * @param thread_data Data for the CPU thread
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Run(ThreadSlice &thread_data) {
    gunrock::app::Iteration_Loop<0, 1, IterationT>(
        thread_data, iterations[thread_data.thread_num]);
    return cudaSuccess;
  }

  /**
   * @brief Enacts a SSSP computing on the specified graph.
   * @param[in] src Source node to start primitive.
   * \return cudaError_t error message(s), if any
   */
  cudaError_t Enact() {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(this->Run_Threads(this));
    util::PrintMsg("GPU SAGE Done.", this->flag & Debug);
    return retval;
  }

  /** @} */
};

}  // namespace sage
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
