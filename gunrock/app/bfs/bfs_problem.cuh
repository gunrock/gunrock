// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bfs_problem.cuh
 *
 * @brief GPU Storage management Structure for BFS Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace bfs {

enum Direction {
  FORWARD = 0,
  BACKWARD = 1,
  UNDECIDED = 2,
};

/**
 * @brief  Speciflying parameters for BFS Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  GUARD_CU(parameters.Use<bool>(
      "mark-pred",
      util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
      false, "Whether to mark predecessor info.", __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Breadth-First Search Problem structure
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sssp
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
  typedef typename GraphT::CscT CscT;
  typedef typename GraphT::GpT GpT;
  typedef _LabelT LabelT;
  typedef _ValueT ValueT;
  typedef unsigned char MaskT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures
  /**
   * @brief Data slice structure containing BFS-specific data on indiviual GPU
   */
  struct DataSlice : BaseDataSlice {
    // util::Array1D<SizeT, VertexT> original_vertex;
    util::Array1D<SizeT, LabelT>
        labels;  // labels to mark latest iteration the vertex been visited
    util::Array1D<SizeT, VertexT> preds;       // predecessors of vertices
    util::Array1D<SizeT, VertexT> temp_preds;  // predecessors of vertices
    util::Array1D<SizeT, SizeT> vertex_markers[2];
    util::Array1D<SizeT, VertexT> unvisited_vertices[2];
    util::Array1D<SizeT, SizeT, util::PINNED> split_lengths;
    util::Array1D<SizeT, VertexT> local_vertices;
    util::Array1D<SizeT, MaskT> visited_masks;
    util::Array1D<SizeT, MaskT> old_mask;
    util::Array1D<SizeT, MaskT *> in_masks;
    util::Array1D<SizeT, Direction> direction_votes;

    SizeT num_visited_vertices, num_unvisited_vertices;
    bool been_in_backward;
    Direction current_direction, previous_direction;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      // original_vertex        .SetName("original_vertex"      );
      labels.SetName("labels");
      preds.SetName("preds");
      temp_preds.SetName("temp_preds");
      vertex_markers[0].SetName("vertex_markers[0]");
      vertex_markers[1].SetName("vertex_markers[1]");
      unvisited_vertices[0].SetName("unvisited_vertices[0]");
      unvisited_vertices[1].SetName("unvisited_vertices[1]");
      local_vertices.SetName("local_vertices");
      split_lengths.SetName("split_length");
      direction_votes.SetName("direction_votes");
      visited_masks.SetName("visited_masks");
      old_mask.SetName("old_mask");
      in_masks.SetName("in_masks");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      // GUARD_CU(original_vertex      .Release(target));
      GUARD_CU(labels.Release(target));
      GUARD_CU(preds.Release(target));
      GUARD_CU(temp_preds.Release(target));
      GUARD_CU(vertex_markers[0].Release(target));
      GUARD_CU(vertex_markers[1].Release(target));
      GUARD_CU(unvisited_vertices[0].Release(target));
      GUARD_CU(unvisited_vertices[1].Release(target));
      GUARD_CU(split_lengths.Release(target));
      GUARD_CU(local_vertices.Release(target));
      GUARD_CU(direction_votes.Release(target));
      GUARD_CU(visited_masks.Release(target));
      GUARD_CU(old_mask.Release(target));
      GUARD_CU(in_masks.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing sssp-specific data on each gpu
     * @param     sub_graph   Sub graph on the GPU.
     * @param[in] num_gpus    Number of GPUs
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

      GUARD_CU(labels.Allocate(sub_graph.nodes, target));
      if (flag & Mark_Predecessors) {
        GUARD_CU(preds.Allocate(sub_graph.nodes, target));
        // GUARD_CU(temp_preds .Allocate(sub_graph.nodes, target));
      }

      GUARD_CU(unvisited_vertices[0].Allocate(sub_graph.nodes, target));
      GUARD_CU(unvisited_vertices[1].Allocate(sub_graph.nodes, target));
      GUARD_CU(split_lengths.Allocate(2, util::HOST | target));
      GUARD_CU(direction_votes.Allocate(4, util::HOST));

      if (flag & Enable_Idempotence) {
        GUARD_CU(visited_masks.Allocate(
            sub_graph.nodes / (sizeof(MaskT) * 8) + 2 * sizeof(VertexT),
            target));
      }

      if (num_gpus > 1) {
        /*if (flag & Mark_Predecessors)
        {
            this->vertex_associate_orgs[0] = preds.GetPointer(target);
            if (!keep_node_num)
            {
                original_vertex.SetPointer(
                    graph_slice->original_vertex.GetPointer(target),
                    graph_slice->original_vertex.GetSize(), target);
            }
        }

        GUARD_CU(this->vertex_associate_orgs.Move(util::HOST, target));
        */

        SizeT local_counter = 0;
        for (VertexT v = 0; v < sub_graph.nodes; v++)
          if (sub_graph.GpT::partition_table[v] == 0) local_counter++;
        GUARD_CU(local_vertices.Allocate(local_counter, util::HOST | target));

        local_counter = 0;
        for (VertexT v = 0; v < sub_graph.nodes; v++) {
          if (sub_graph.GpT::partition_table[v] == 0) {
            local_vertices[local_counter] = v;
            local_counter++;
          }
        }
        GUARD_CU(local_vertices.Move(util::HOST, target));
      }

      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // end of Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] src      Source vertex to start.
     * @param[in] location Memory location to work on
     * \return cudaError_t Error message(s), if any
     */
    cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;

      SizeT nodes = this->sub_graph->nodes;
      num_visited_vertices = 0;
      num_unvisited_vertices = 0;
      been_in_backward = false;
      current_direction = FORWARD;
      previous_direction = FORWARD;

      GUARD_CU(util::SetDevice(this->gpu_idx));
      for (int i = 0; i < 4; i++) direction_votes[i] = UNDECIDED;

      // Allocate output labels if necessary
      GUARD_CU(labels.EnsureSize_(nodes, target));
      GUARD_CU(labels.ForEach(
          [] __host__ __device__(LabelT & label) {
            label = util::PreDefinedValues<LabelT>::MaxValue;
          },
          nodes, target, this->stream));

      if (this->flag & Mark_Predecessors) {
        // Allocate preds if necessary
        GUARD_CU(preds.EnsureSize_(nodes, target));
        GUARD_CU(preds.ForEach(
            [] __host__ __device__(VertexT & pred) {
              pred = util::PreDefinedValues<VertexT>::InvalidValue;
            },
            nodes, target, this->stream));
      }

      if (this->flag & Enable_Idempotence) {
        GUARD_CU(visited_masks.ForEach(
            [] __host__ __device__(MaskT & mask) { mask = 0; },
            util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));
      }

      return retval;
    }  // end of Reset
  };   // end of DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief BFSProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief BFSProblem default destructor
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
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      GUARD_CU(data_slices[gpu].Release(target));
    }

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
   * @brief Copy result labels and/or predecessors computed on the GPU back to
   *host-side vectors.
   * @param[out] h_labels Host array to store computed vertex labels
   * @param[out] h_preds  Host array to store computed vertex predecessors
   * @param[in]  target where the results are stored
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(LabelT *h_labels, VertexT *h_preds = NULL,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;

    if (this->num_gpus == 1) {
      auto &data_slice = data_slices[0][0];

      if (target == util::DEVICE) {
        // Set device
        GUARD_CU(util::SetDevice(this->gpu_idx[0]));

        GUARD_CU(data_slice.labels.SetPointer(h_labels, nodes, util::HOST));
        GUARD_CU(data_slice.labels.Move(util::DEVICE, util::HOST));

        if (this->flag & Mark_Predecessors) {
          GUARD_CU(data_slice.preds.SetPointer(h_preds, nodes, util::HOST));
          GUARD_CU(data_slice.preds.Move(util::DEVICE, util::HOST));
        }
      }

      else if (target == util::HOST) {
        GUARD_CU(data_slice.labels.ForAll(
            [h_labels] __host__ __device__(const LabelT *labels,
                                           const VertexT &v) {
              h_labels[v] = labels[v];
            },
            nodes, util::HOST));

        if (this->flag & Mark_Predecessors)
          GUARD_CU(data_slice.preds.ForAll(
              [h_preds] __host__ __device__(const VertexT *preds,
                                            const VertexT &v) {
                h_preds[v] = preds[v];
              },
              nodes, util::HOST));
      }
    }

    else {  // num_gpus != 1
      util::Array1D<SizeT, LabelT *> th_labels;
      util::Array1D<SizeT, VertexT *> th_preds;
      th_labels.SetName("bfs::Problem::Extract::th_labels");
      th_preds.SetName("bfs::Problem::Extract::th_preds");
      GUARD_CU(th_labels.Allocate(this->num_gpus, util::HOST));
      GUARD_CU(th_preds.Allocate(this->num_gpus, util::HOST));

      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        auto &data_slice = data_slices[gpu][0];
        if (target == util::DEVICE) {
          GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
          GUARD_CU(data_slice.labels.Move(util::DEVICE, util::HOST));
          if (this->flag & Mark_Predecessors)
            GUARD_CU(data_slice.preds.Move(util::DEVICE, util::HOST));
        }
        th_labels[gpu] = data_slice.labels.GetPointer(util::HOST);
        th_preds[gpu] = data_slice.preds.GetPointer(util::HOST);
      }  // end for(gpu)

      for (VertexT v = 0; v < nodes; v++) {
        int gpu = this->org_graph->GpT::partition_table[v];
        VertexT v_ = v;
        if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) == 0)
          v_ = this->org_graph->GpT::convertion_table[v];

        h_labels[v] = th_labels[gpu][v_];
        if (this->flag & Mark_Predecessors) h_preds[v] = th_preds[gpu][v_];
      }

      GUARD_CU(th_labels.Release());
      GUARD_CU(th_preds.Release());
    }  // end if (num_gpus ==1)

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that SSSP processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    if (this->parameters.template Get<bool>("mark-pred"))
      this->flag = this->flag | Mark_Predecessors;
    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));
      auto &data_slice = data_slices[gpu][0];
      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));
    }  // end for(gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(VertexT src, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));
      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    // Fillin the initial input_queue for BFS problem
    int gpu;
    VertexT src_;
    if (this->num_gpus <= 1) {
      gpu = 0;
      src_ = src;
    } else {
      gpu = this->org_graph->partition_table[src];
      if (this->flag & partitioner::Keep_Node_Num)
        src_ = src;
      else
        src_ = this->org_graph->GpT::convertion_table[src];
    }

    if (target & util::HOST) {
      data_slices[gpu]->labels[src_] = 0;
      if (this->flag & Mark_Predecessors)
        data_slices[gpu]->preds[src_] =
            util::PreDefinedValues<VertexT>::InvalidValue;
      if (this->flag & Enable_Idempotence) {
        VertexT mask_pos = src_ / (8 * sizeof(MaskT));
        data_slices[gpu]->visited_masks[mask_pos] =
            1 << (src_ % (8 * sizeof(MaskT)));
      }
    }

    if (target & util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
      GUARD_CU(data_slices[gpu]->labels.ForAll(
          [src_] __host__ __device__(LabelT * labels, const SizeT &v) {
            labels[src_] = 0;
          },
          1, util::DEVICE));

      if (this->flag & Mark_Predecessors) {
        GUARD_CU(data_slices[gpu]->preds.ForAll(
            [src_] __host__ __device__(VertexT * preds, const SizeT &v) {
              preds[src_] = util::PreDefinedValues<VertexT>::InvalidValue;
            },
            1, util::DEVICE));
      }

      if (this->flag & Enable_Idempotence) {
        VertexT mask_pos = src_ / (8 * sizeof(MaskT));
        GUARD_CU(data_slices[gpu]->visited_masks.ForAll(
            [mask_pos, src_] __host__ __device__(MaskT * masks,
                                                 const SizeT &v) {
              masks[mask_pos] = 1 << (src_ % (8 * sizeof(MaskT)));
            },
            1, util::DEVICE));
      }
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }  // end of reset

  /** @} */

};  // end of problem

}  // namespace bfs
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
