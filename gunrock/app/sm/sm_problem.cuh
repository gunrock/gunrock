// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sm_problem.cuh
 *
 * @brief GPU Storage management Structure for SM Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/sm/sm_test.cuh>

namespace gunrock {
namespace app {
namespace sm {

/**
 * @brief Speciflying parameters for SM Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief Subgraph Matching Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sm
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT, typename _LabelT = typename _GraphT::VertexT,
          typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::CsrT CsrT;
  typedef typename GraphT::GpT GpT;
  typedef _LabelT LabelT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures

  /**
   * @brief Data structure containing SM-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // sm-specific storage arrays
    util::Array1D<SizeT, VertexT> subgraphs;     // number of subgraphs
    util::Array1D<SizeT, VertexT> query_labels;  // query graph labels
    util::Array1D<SizeT, VertexT> data_labels;   // data graph labels

    util::Array1D<SizeT, SizeT> query_ro;     // query graph row offsets
    util::Array1D<SizeT, VertexT> query_ci;   // query graph column indices
    util::Array1D<SizeT, SizeT> data_degree;  // data graph nodes' degrees
    util::Array1D<SizeT, bool> isValid; /** < Used for data node validation */
    util::Array1D<SizeT, SizeT>
        counter; /** < Used for minimum degree in query graph */
    util::Array1D<SizeT, SizeT> num_subs; /** < Used for counting subgraphs   */
    util::Array1D<SizeT, SizeT> results;  /** < Used for gpu results   */
    util::Array1D<SizeT, SizeT>
        constrain;                    /** < Smallest degree in query graph   */
    util::Array1D<SizeT, VertexT> NG; /** < Used for query node explore seq  */
    util::Array1D<SizeT, VertexT>
        NG_src; /** < Used for query node sequence non-tree edge info */
    util::Array1D<SizeT, VertexT>
        NG_dest; /** < Used for query node sequence non-tree edge info */
    util::Array1D<SizeT, VertexT>
        partial; /** < Used for storing partial results */
    util::Array1D<SizeT, bool>
        flags; /** < Used for storing compacted src nodes */
    util::Array1D<SizeT, VertexT>
        indices;       /** < Used for storing intermediate flag val */
    SizeT nodes_query; /** < Used for number of query nodes */
    SizeT num_matches; /** < Used for number of matches in the result */

    // query graph col_indices
    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      subgraphs.SetName("subgraphs");
      query_labels.SetName("query_labels");
      data_labels.SetName("data_labels");
      query_ro.SetName("query_ro");
      data_degree.SetName("data_degree");
      isValid.SetName("isValid");
      counter.SetName("counter");
      num_subs.SetName("num_subs");
      results.SetName("results");
      constrain.SetName("constrain");
      NG.SetName("NG");
      NG_src.SetName("NG_src");
      NG_dest.SetName("NG_dest");
      partial.SetName("partial");
      flags.SetName("src_node_id");
      indices.SetName("indices");
      nodes_query = 0;
      num_matches = 0;
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

      GUARD_CU(subgraphs.Release(target));
      GUARD_CU(query_labels.Release(target));
      GUARD_CU(data_labels.Release(target));
      GUARD_CU(data_degree.Release(target));
      GUARD_CU(isValid.Release(target));
      GUARD_CU(counter.Release(target));
      GUARD_CU(num_subs.Release(target));
      GUARD_CU(results.Release(target));
      GUARD_CU(constrain.Release(target));
      GUARD_CU(NG.Release(target));
      GUARD_CU(NG_src.Release(target));
      GUARD_CU(NG_dest.Release(target));
      GUARD_CU(partial.Release(target));
      GUARD_CU(flags.Release(target));
      GUARD_CU(indices.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing sm-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
     * @param[in] gpu_idx     GPU device index
     * @param[in] target      Targeting device location
     * @param[in] flag        Problem flag containling options
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Init(GraphT &sub_graph, GraphT &data_graph, GraphT &query_graph,
                     int num_gpus = 1, int gpu_idx = 0,
                     util::Location target = util::DEVICE,
                     ProblemFlag flag = Problem_None) {
      cudaError_t retval = cudaSuccess;
      int num_query_edge = query_graph.edges / 2;
      int num_query_node = query_graph.nodes;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      GUARD_CU(subgraphs.Allocate(sub_graph.nodes, target));
      GUARD_CU(
          query_ro.Allocate(num_query_node + 1, util::HOST | util::DEVICE));
      GUARD_CU(data_degree.Allocate(data_graph.nodes, util::DEVICE));
      GUARD_CU(isValid.Allocate(data_graph.nodes, util::DEVICE));
      GUARD_CU(counter.Allocate(1, util::HOST | util::DEVICE));
      GUARD_CU(num_subs.Allocate(1, util::HOST | util::DEVICE));
      GUARD_CU(results.Allocate(data_graph.nodes, util::HOST | util::DEVICE));
      GUARD_CU(constrain.Allocate(1, util::HOST | util::DEVICE));
      GUARD_CU(NG.Allocate(2 * num_query_node, util::HOST | util::DEVICE));
      // non-tree edges only exist in the following condition; double the size
      // to store nodes for duplicate edges
      if (num_query_edge - query_graph.nodes + 1 > 0) {
        GUARD_CU(NG_src.Allocate((num_query_edge - num_query_node + 1) * 2,
                                 util::HOST | util::DEVICE));
        GUARD_CU(NG_dest.Allocate((num_query_edge - num_query_node + 1) * 2,
                                  util::HOST | util::DEVICE));
      }
      // partial results storage: as much as possible
      GUARD_CU(
          partial.Allocate(num_query_node * data_graph.edges, util::DEVICE));
      GUARD_CU(flags.Allocate(data_graph.nodes, util::DEVICE));
      GUARD_CU(indices.Allocate(data_graph.nodes, util::DEVICE));

      // Initialize query graph node degree by row offsets
      // neighbor node encoding = sum of neighbor node labels
      int *query_degree = new int[num_query_node];
      for (int i = 0; i < num_query_node; i++) {
        query_degree[i] =
            query_graph.row_offsets[i + 1] - query_graph.row_offsets[i];
      }
      // Generate query graph node exploration sequence based on maximum
      // likelihood estimation (MLE) node mapping degree, TODO:probablity
      // estimation based on label and degree
      int *d_m = new int[num_query_node];
      memset(d_m, 0, sizeof(int) * num_query_node);
      int degree_max = query_degree[0];
      int degree_min = query_degree[0];
      int index = 0;
      for (int i = 0; i < num_query_node; ++i) {
        if (i == 0) {
          // find maximum degree in query graph and corresponding node index
          for (int j = 1; j < num_query_node; ++j) {
            if (query_degree[j] > degree_max) {
              index = j;
              degree_max = query_degree[j];
            } else {
              degree_min = query_degree[j];
            }
          }
        } else {
          int dm_max = 0;
          index = 0;
          while (d_m[index] == -1) {
            index++;
          }
          for (int j = 0; j < num_query_node; ++j) {
            if (d_m[j] >= 0) {
              if (index * degree_max + query_degree[j] > dm_max) {
                dm_max = index * degree_max + query_degree[j];
                index = j;
              }
            }
          }
        }
        // put the node index with max degree in NG, and mark that node as
        // solved -1, and add priority to its neighbors
        NG[i] = index;
        d_m[index] = -1;
        for (int j = query_graph.row_offsets[index];
             j < query_graph.row_offsets[index + 1]; ++j)
          if (d_m[query_graph.column_indices[j]] != -1)
            d_m[query_graph.column_indices[j]]++;
      }
      constrain[0] = degree_min;
      delete[] d_m;
      delete[] query_degree;

      bool *node_visited = new bool[num_query_node];
      bool *edge_visited = new bool[num_query_node * num_query_node];
      memset(node_visited, false, sizeof(bool) * num_query_node);
      memset(edge_visited, false,
             sizeof(bool) * num_query_node * num_query_node);
      // fill in query graph non-tree edges info for each node
      if (num_query_edge - num_query_node + 1 > 0) {
        int i = 0;
        int j = 0;
        for (int id = 0; id < num_query_node; ++id) {
          int src = NG[id];
          node_visited[src] = true;
          for (int idx = query_graph.row_offsets[src];
               idx < query_graph.row_offsets[src + 1]; ++idx) {
            int dest = query_graph.column_indices[idx];
            if (edge_visited[src * num_query_node + dest]) continue;
            edge_visited[src * num_query_node + dest] = true;
            edge_visited[dest * num_query_node + src] = true;
            if (node_visited[dest]) {
              // This is a non-tree edge, store src and dest
              NG_src[i++] = src;
              NG_src[i++] = dest;
              NG_dest[j++] = dest;
              NG_dest[j++] = src;
            }
            node_visited[dest] = true;
          }
        }
      }
      delete[] node_visited;
      delete[] edge_visited;

      // Add 1 look ahead info: each query node's neighbors' min degree in the
      // pos of (query_node_id, min_degree) rearange NG pos
      for (int i = query_graph.nodes - 1; i >= 0; --i) {
        NG[i * 2] = NG[i];
      }
      for (int i = 0; i < query_graph.nodes; ++i) {
        VertexT src = NG[i * 2];
        int min_degree = INT_MAX;
        for (int j = query_graph.row_offsets[src];
             j < query_graph.row_offsets[src + 1]; ++j) {
          VertexT dest = query_graph.column_indices[src];
          if (query_graph.row_offsets[dest + 1] -
                  query_graph.row_offsets[dest] <
              min_degree) {
            min_degree = query_graph.row_offsets[dest + 1] -
                         query_graph.row_offsets[dest];
          }
        }
        NG[i * 2 + 1] = min_degree;
      }

      /*            std::cout << "==============NG===========" << std::endl;
                  for (int i = 0; i < 2 * query_graph.nodes; ++i) {
                      std::cout << NG[i] << std::endl;
                  }
                  std::cout << "==============NG_src===========" << std::endl;
                  for (int i = 0; i < num_query_edge - query_graph.nodes + 1;
         ++i) { std::cout << NG_src[i] << std::endl;
                  }
                  std::cout << "==============NG_dest===========" << std::endl;
                  for (int i = 0; i < num_query_edge - query_graph.nodes + 1;
         ++i) { std::cout << NG_dest[i] << std::endl;
                  }
                  std::cout << "==============MIN degree===========" <<
         std::endl; std::cout << constrain[0] << std::endl;
      */
      GUARD_CU(NG.Move(util::HOST, target));
      GUARD_CU(NG_src.Move(util::HOST, target));
      GUARD_CU(NG_dest.Move(util::HOST, target));
      GUARD_CU(constrain.Move(util::HOST, target));
      // Initialize query row offsets with query_graph.row_offsets
      GUARD_CU(query_ro.ForAll(
          [query_graph] __host__ __device__(SizeT * x, const SizeT &pos) {
            x[pos] = query_graph.row_offsets[pos];
          },
          query_graph.nodes + 1, util::HOST));
      GUARD_CU(query_ro.Move(util::HOST, target));
      GUARD_CU(isValid.ForAll(
          [] __device__(bool *x, const SizeT &pos) { x[pos] = true; },
          data_graph.nodes, target, this->stream));
      GUARD_CU(flags.ForAll(
          [] __device__(bool *x, const SizeT &pos) { x[pos] = false; },
          data_graph.nodes, target, this->stream));
      GUARD_CU(indices.ForAll(
          [] __device__(VertexT * x, const SizeT &pos) { x[pos] = pos; },
          data_graph.nodes, target, this->stream));
      GUARD_CU(data_degree.ForAll(
          [] __device__(SizeT * x, const SizeT &pos) { x[pos] = 0; },
          data_graph.nodes, target, this->stream));
      GUARD_CU(counter.ForAll(
          [] __device__(SizeT * x, const SizeT &pos) { x[pos] = 0; }, 1, target,
          this->stream));
      GUARD_CU(num_subs.ForAll(
          [] __host__ __device__(SizeT * x, const SizeT &pos) { x[pos] = 0; },
          1, target, this->stream));
      GUARD_CU(results.ForAll(
          [] __host__ __device__(SizeT * x, const SizeT &pos) { x[pos] = 0; },
          data_graph.nodes, target, this->stream));

      nodes_query = query_graph.nodes;

      if (target & util::DEVICE) {
        GUARD_CU(sub_graph.CsrT::Move(util::HOST, target, this->stream));
        return retval;
      }

      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;
      SizeT num_nodes = this->sub_graph->nodes;
      SizeT num_edges = this->sub_graph->edges;

      // Ensure data are allocated
      GUARD_CU(subgraphs.EnsureSize_(num_nodes, target));
      //            GUARD_CU(nodes     .EnsureSize_(num_nodes, target));

      // Reset data
      GUARD_CU(subgraphs.ForEach(
          [] __host__ __device__(VertexT & x) { x = (VertexT)0; }, num_nodes,
          target, this->stream));

      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief SMProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief SMProblem default destructor
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
   * the source.
   * @param[out] h_preds     Host array to store computed vertex predecessors.
   * @param[in]  target where the results are stored
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(VertexT *h_subgraphs,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      // Set device
      if (target == util::DEVICE) {
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(data_slice.results.SetPointer(h_subgraphs, nodes, util::HOST));
        GUARD_CU(data_slice.results.Move(util::DEVICE, util::HOST));
      } else if (target == util::HOST) {
        GUARD_CU(data_slice.results.ForEach(
            h_subgraphs,
            [] __host__ __device__(const VertexT &d_x, VertexT &h_x) {
              h_x = d_x;
            },
            nodes, util::HOST));
      }
    } else {  // num_gpus != 1

      // !! MultiGPU not implemented

      // util::Array1D<SizeT, ValueT *> th_subgraphs;
      // util::Array1D<SizeT, VertexT*> th_nodes;
      // th_subgraphs.SetName("bfs::Problem::Extract::th_subgraphs");
      // th_nodes     .SetName("bfs::Problem::Extract::th_nodes");
      // GUARD_CU(th_subgraphs.Allocate(this->num_gpus, util::HOST));
      // GUARD_CU(th_nodes    .Allocate(this->num_gpus, util::HOST));

      // for (int gpu = 0; gpu < this->num_gpus; gpu++)
      // {
      //     auto &data_slice = data_slices[gpu][0];
      //     if (target == util::DEVICE)
      //     {
      //         GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      //         GUARD_CU(data_slice.subgraphs.Move(util::DEVICE, util::HOST));
      //         GUARD_CU(data_slice.nodes.Move(util::DEVICE, util::HOST));
      //     }
      //     th_subgraphs[gpu] = data_slice.subgraphs.GetPointer(util::HOST);
      //     th_nodes    [gpu] = data_slice.nodes    .GetPointer(util::HOST);
      // } //end for(gpu)

      // for (VertexT v = 0; v < nodes; v++)
      // {
      //     int gpu = this -> org_graph -> GpT::partition_table[v];
      //     VertexT v_ = v;
      //     if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) != 0)
      //         v_ = this -> org_graph -> GpT::convertion_table[v];

      //     h_subgraphs[v] = th_subgraphs[gpu][v_];
      //     h_node      [v] = th_nodes     [gpu][v_];
      // }

      // GUARD_CU(th_subgraphs.Release());
      // GUARD_CU(th_nodes     .Release());
    }  // end if

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SM processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &data_graph, GraphT &query_graph,
                   util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(data_graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], data_graph, query_graph,
                               this->num_gpus, this->gpu_idx[gpu], target,
                               this->flag));
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

}  // namespace sm
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
