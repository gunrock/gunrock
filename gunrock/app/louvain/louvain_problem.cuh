// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * template_problem.cuh
 *
 * @brief GPU Storage management Structure for Louvain Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/louvain/louvain_test.cuh>

namespace gunrock {
namespace app {
namespace louvain {

/**
 * @brief Speciflying parameters for Louvain Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief Louvain Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename util::PreDefinedValues<VertexT>::PromoteType EdgePairT;
  static const int BITS_VERTEXT = sizeof(VertexT) * 8;
  static const EdgePairT VERTEXT_MASK =
      (EdgePairT)util::PreDefinedValues<VertexT>::AllOnes;
  typedef _ValueT ValueT;

  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // COnverters
  static __host__ __device__ __forceinline__ EdgePairT
  MakePair(const VertexT &first, const VertexT &second) {
    return (((EdgePairT)first) << BITS_VERTEXT) + second;
  }

  static __host__ __device__ __forceinline__ VertexT
  GetFirst(const EdgePairT &pair) {
    return pair >> BITS_VERTEXT;
  }

  static __host__ __device__ __forceinline__ VertexT
  GetSecond(const EdgePairT &pair) {
    return pair & VERTEXT_MASK;
  }

  // Helper structures

  /**
   * @brief Data structure containing Louvain-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // communities the vertices are current in
    util::Array1D<SizeT, VertexT> current_communities;

    // communities to move in
    util::Array1D<SizeT, VertexT> next_communities;

    // size of communities
    util::Array1D<SizeT, VertexT> community_sizes;

    // sum of edge weights from vertices
    util::Array1D<SizeT, ValueT> w_v2;

    // sum of edge weights from vertices to self
    util::Array1D<SizeT, ValueT> w_v2self;

    // sum of edge weights from communities
    util::Array1D<SizeT, ValueT> w_c2;

    // communities each edge belongs to
    // util::Array1D<SizeT, VertexT>   edge_comms0;
    // util::Array1D<SizeT, VertexT>   edge_comms1;

    // weights of edges
    util::Array1D<SizeT, ValueT> edge_weights0;
    util::Array1D<SizeT, ValueT> edge_weights1;

    // segment offsets
    util::Array1D<SizeT, SizeT> seg_offsets0;
    util::Array1D<SizeT, SizeT> seg_offsets1;

    // edge pairs for sorting
    util::Array1D<SizeT, EdgePairT> edge_pairs0;
    util::Array1D<SizeT, EdgePairT> edge_pairs1;

    // temp space for cub
    util::Array1D<uint64_t, char> cub_temp_space;

    // number of neighbor communities
    util::Array1D<SizeT, SizeT> num_neighbor_comms;

    // Number of new communities
    util::Array1D<SizeT, SizeT> num_new_comms;

    // Number of new edges
    util::Array1D<SizeT, SizeT> num_new_edges;

    // base of modularity grain
    util::Array1D<SizeT, ValueT> gain_bases;

    // gain of each moves
    util::Array1D<SizeT, ValueT> max_gains;

    // gain from current iteration
    util::Array1D<SizeT, ValueT> iter_gain;

    // gain from current pass
    ValueT pass_gain;

    // sum of edge weights
    ValueT m2;

    // modularity
    ValueT q;

    // Contracted graph
    GraphT new_graphs[2];

    // std::vector<util::Array1D<SizeT, VertexT>*> pass_communities;
    util::Array1D<SizeT, VertexT> *pass_communities;
    int num_pass, max_iters;

    // Whether to use cubRedixSort instead of cubSegmentRadixSort
    // bool unify_segments;
    /*
     * @brief Default constructor
     */
    DataSlice()
        : BaseDataSlice()
    // new_graph    (NULL),
    // unify_segments(false)
    {
      current_communities.SetName("current_communities");
      next_communities.SetName("next_communities");
      community_sizes.SetName("community_sizes");
      w_v2.SetName("w_v2");
      w_v2self.SetName("w_v2self");
      w_c2.SetName("w_c2");
      // edge_comms0        .SetName("edge_comms0"        );
      // edge_comms1        .SetName("edge_comms1"        );
      edge_weights0.SetName("edge_weights0");
      edge_weights1.SetName("edge_weights1");
      seg_offsets0.SetName("seg_offsets0");
      seg_offsets1.SetName("seg_offsets1");
      edge_pairs0.SetName("edge_pairs0");
      edge_pairs1.SetName("edge_pairs1");
      cub_temp_space.SetName("cub_temp_space");
      num_neighbor_comms.SetName("num_neighbor_comms");
      num_new_comms.SetName("num_new_comms");
      num_new_edges.SetName("num_new_edges");
      gain_bases.SetName("gain_bases");
      max_gains.SetName("max_gains");
      iter_gain.SetName("iter_gain");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    /*
     * @brief Releasing allocated memory space
     * @param[in] target      The location to release memory from
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      GUARD_CU(current_communities.Release(target));
      GUARD_CU(next_communities.Release(target));
      GUARD_CU(community_sizes.Release(target));
      GUARD_CU(w_v2.Release(target));
      GUARD_CU(w_v2self.Release(target));
      GUARD_CU(w_c2.Release(target));

      // GUARD_CU(edge_comms0        .Release(target));
      // GUARD_CU(edge_comms1        .Release(target));
      GUARD_CU(edge_weights0.Release(target));
      GUARD_CU(edge_weights1.Release(target));
      GUARD_CU(seg_offsets0.Release(target));
      GUARD_CU(seg_offsets1.Release(target));
      GUARD_CU(edge_pairs0.Release(target));
      GUARD_CU(edge_pairs1.Release(target));

      GUARD_CU(cub_temp_space.Release(target));
      GUARD_CU(num_neighbor_comms.Release(target));
      GUARD_CU(num_new_comms.Release(target));
      GUARD_CU(num_new_edges.Release(target));
      GUARD_CU(gain_bases.Release(target));
      GUARD_CU(max_gains.Release(target));
      GUARD_CU(iter_gain.Release(target));

      // if (new_graph != NULL)
      {
        GUARD_CU(new_graphs[0].Release(target));
        GUARD_CU(new_graphs[1].Release(target));
        // delete new_graph; new_graph = NULL;
      }

      // for (auto &pass_comm : pass_communities)
      if (pass_communities != NULL)
        for (int i = 0; i < max_iters; i++) {
          auto &pass_comm = pass_communities[i];
          // if (pass_comm == NULL)
          //    continue;
          GUARD_CU(pass_comm.Release(target));
          // delete pass_comm; pass_comm = NULL;
        }
      /// pass_communities.clear();
      delete[] pass_communities;
      pass_communities = NULL;

      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing Louvain-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));

      GUARD_CU(current_communities.Allocate(sub_graph.nodes, target));
      GUARD_CU(next_communities.Allocate(sub_graph.nodes, target));
      GUARD_CU(community_sizes.Allocate(sub_graph.nodes, target));
      GUARD_CU(w_v2.Allocate(sub_graph.nodes, target));
      GUARD_CU(w_v2self.Allocate(sub_graph.nodes, target));
      GUARD_CU(w_c2.Allocate(sub_graph.nodes, target));

      // GUARD_CU(edge_comms0        .Allocate(sub_graph.edges+1, target));
      // GUARD_CU(edge_comms1        .Allocate(sub_graph.edges+1, target));
      GUARD_CU(edge_weights0.Allocate(sub_graph.edges + 1, target));
      GUARD_CU(edge_weights1.Allocate(sub_graph.edges + 1, target));
      GUARD_CU(seg_offsets0.Allocate(sub_graph.edges + 1, target));
      GUARD_CU(seg_offsets1.Allocate(sub_graph.edges + 1, target));
      GUARD_CU(edge_pairs0.Allocate(sub_graph.edges + 1, target));
      GUARD_CU(edge_pairs1.Allocate(sub_graph.edges + 1, target));

      GUARD_CU(num_neighbor_comms.Allocate(1, target | util::HOST));
      GUARD_CU(num_new_comms.Allocate(1, target | util::HOST));
      GUARD_CU(num_new_edges.Allocate(1, target | util::HOST));
      GUARD_CU(cub_temp_space.Allocate(1, target));
      GUARD_CU(gain_bases.Allocate(sub_graph.nodes, target));
      GUARD_CU(max_gains.Allocate(sub_graph.nodes, target));
      GUARD_CU(iter_gain.Allocate(1, target | util::HOST));

      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      }
      pass_communities = new util::Array1D<SizeT, VertexT>[max_iters + 1];
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;

      // if (new_graph != NULL)
      {
        // GUARD_CU(new_graphs[0].Release(target));
        // GUARD_CU(new_graphs[1].Release(target));
        // delete new_graph; new_graph = NULL;
      }

      pass_gain = 0;

      // for (auto &pass_comm : pass_communities)
      // for (int i = 0; i < max_iters; i++)
      //{
      //    auto &pass_comm = pass_communities[i];
      //    if (pass_comm == NULL)
      //        continue;
      //    GUARD_CU(pass_comm -> Release(target));
      //    delete pass_comm; pass_comm = NULL;
      //}
      num_pass = 0;
      // pass_communities.clear();

      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief LouvainProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief LouvainProblem default destructor
   */
  virtual ~Problem() { Release(); }

  /*
   * @brief Releasing allocated memory space
   * @param[in] target      The location to release memory from
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Release(util::Location target = util::LOCATION_ALL) {
    cudaError_t retval = cudaSuccess;
    if (data_slices == NULL) return retval;
    for (int i = 0; i < this->num_gpus; i++)
      GUARD_CU(data_slices[i].Release(target));

    if ((target & util::HOST) != 0 &&
        data_slices[0].GetPointer(util::DEVICE) == NULL) {
      delete[] data_slices;
      data_slices = NULL;
    }
    GUARD_CU(BaseProblem::Release(target));
    return retval;
  }

  /**
   * \addtogroup PublicInterface
   * @{
   */

  /**
   * @brief Copy result distancess computed on GPUs back to host-side arrays.
   * @param[out] h_distances Host array to store computed vertex distances from
   * the source. \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(
      VertexT *h_communities,
      std::vector<std::vector<VertexT> *> *pass_communities = NULL,
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    bool has_pass_communities = false;
    if (pass_communities != NULL)
      has_pass_communities = true;
    else
      pass_communities = new std::vector<std::vector<VertexT> *>;

    if (this->num_gpus == 1) {
      for (VertexT v = 0; v < nodes; v++) h_communities[v] = v;
      auto &data_slice = data_slices[0][0];

      // for (auto &pass_comm : data_slice.pass_communities)
      for (int i = 0; i <= data_slice.num_pass; i++) {
        auto &v2c = data_slice.pass_communities[i];
        for (VertexT v = 0; v < nodes; v++)
          h_communities[v] = v2c[h_communities[v]];
      }
    } else {  // num_gpus != 1
      // TODO: extract the results from multiple GPUs, e.g.:
      // util::Array1D<SizeT, ValueT *> th_distances;
      // th_distances.SetName("bfs::Problem::Extract::th_distances");
      // GUARD_CU(th_distances.Allocate(this->num_gpus, util::HOST));

      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        auto &data_slice = data_slices[gpu][0];
        if (target == util::DEVICE) {
          GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
          // GUARD_CU(data_slice.distances.Move(util::DEVICE, util::HOST));
        }
        // th_distances[gpu] = data_slice.distances.GetPointer(util::HOST);
      }  // end for(gpu)

      for (VertexT v = 0; v < nodes; v++) {
        int gpu = this->org_graph->GpT::partition_table[v];
        VertexT v_ = v;
        if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
          v_ = this->org_graph->GpT::convertion_table[v];

        // h_distances[v] = th_distances[gpu][v_];
      }

      // GUARD_CU(th_distances.Release());
    }  // end if

    // Clearn-up
    if (!has_pass_communities) {
      for (auto it = pass_communities->begin(); it != pass_communities->end();
           it++) {
        (*it)->clear();
        delete *it;
      }
      pass_communities->clear();
      delete pass_communities;
      pass_communities = NULL;
    }

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that Louvain processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    ValueT m2 = 0;
    ValueT q = Get_Modularity<GraphT, ValueT>(graph);

    for (SizeT e = 0; e < graph.edges; e++) m2 += graph.CsrT::edge_values[e];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      // data_slice.unify_segments
      //    = this -> parameters.template Get<bool>("unify-segments");
      data_slice.max_iters = this->parameters.template Get<int>("max-iters");
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));

      data_slice.m2 = m2;
      data_slice.q = q;
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    return retval;
  }

  /** @} */
};

}  // namespace louvain
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
