// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * lp_problem.cuh
 *
 * @brief GPU Storage management Structure for LP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace lp {

/**
 * @brief  Speciflying parameters for LP Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));

  return retval;
}

/**
 * @brief Label Propagation Problem structure
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
   * @brief Data slice structure containing LP-specific data on indiviual GPU
   */
  struct DataSlice : BaseDataSlice {

    util::Array1D<SizeT, LabelT> labels;  // labels (in the current iteration) for each vertex 
    util::Array1D<SizeT, LabelT> old_labels; // labels (in the previous iteration) for each vertex
    util::Array1D<SizeT, SizeT> vertex_markers[2];
    util::Array1D<SizeT, SizeT, util::PINNED> split_lengths;
    util::Array1D<SizeT, VertexT> local_vertices;
    util::Array1D<SizeT, MaskT> visited_masks;
    util::Array1D<SizeT, MaskT> old_mask;
    util::Array1D<SizeT, VertexT> unvisited_vertices[2];
    util::Array1D<SizeT, MaskT *> in_masks;

    
    util::Array1D<SizeT, LabelT> neighbour_labels;
    util::Array1D<SizeT, int> neighbour_labels_size;

    // segments_temp stores the relative segments, and segments stores the (cumulative scan) absolute segments
    util::Array1D<SizeT, int> segments;
    util::Array1D<SizeT, int> segments_temp;
    util::Array1D<SizeT, int> segments_size;

    util::Array1D<uint64_t, char> cub_temp_storage;
    SizeT num_visited_vertices, num_unvisited_vertices;
    bool been_in_backward;

    util::Array1D<SizeT, int> visited;
    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {

      labels.SetName("labels");
      old_labels.SetName("old_labels");
      vertex_markers[0].SetName("vertex_markers[0]");
      vertex_markers[1].SetName("vertex_markers[1]");
      unvisited_vertices[0].SetName("unvisited_vertices[0]");
      unvisited_vertices[1].SetName("unvisited_vertices[1]");
      local_vertices.SetName("local_vertices");
      split_lengths.SetName("split_length");
      visited_masks.SetName("visited_masks");
      old_mask.SetName("old_mask");
      in_masks.SetName("in_masks");
  
      neighbour_labels.SetName("neighbour_labels");
      neighbour_labels_size.SetName("neighbour_labels_size");

      segments.SetName("segments");
      segments_size.SetName("segments_size");
      segments_temp.SetName("segments_temp");
      cub_temp_storage.SetName("cub_temp_storage");

      visited.SetName("visited");
    }

    /*
     * @brief Default destructor
     */
    virtual ~DataSlice() { Release(); }

    cudaError_t Release(util::Location target = util::LOCATION_ALL) {
      cudaError_t retval = cudaSuccess;
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx));

      GUARD_CU(visited.Release(target));
      GUARD_CU(labels.Release(target));
      GUARD_CU(old_labels.Release(target));
      GUARD_CU(vertex_markers[0].Release(target));
      GUARD_CU(vertex_markers[1].Release(target));
      GUARD_CU(split_lengths.Release(target));
      GUARD_CU(local_vertices.Release(target));
      GUARD_CU(visited_masks.Release(target));
      GUARD_CU(unvisited_vertices[0].Release(target));
      GUARD_CU(unvisited_vertices[1].Release(target));
      GUARD_CU(old_mask.Release(target));
      GUARD_CU(in_masks.Release(target));
      GUARD_CU(in_masks.Release(target));
      GUARD_CU(neighbour_labels.Release(target));
      GUARD_CU(neighbour_labels_size.Release(target));
      GUARD_CU(segments_size.Release(target));
      GUARD_CU(segments.Release(target));
      GUARD_CU(segments_temp.Release(target));
      GUARD_CU(cub_temp_storage.Release(target));
      GUARD_CU(BaseDataSlice ::Release(target));
      return retval;
    }

    /**
     * @brief initializing lp-specific data on each gpu
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
      GUARD_CU(old_labels.Allocate(sub_graph.nodes, target));
      GUARD_CU(segments.Allocate(sub_graph.nodes, target));
      GUARD_CU(segments_temp.Allocate(sub_graph.nodes, target));
      GUARD_CU(segments_size.Allocate(1, target));
      GUARD_CU(neighbour_labels_size.Allocate(1,  util::DEVICE | util::HOST));
      GUARD_CU(cub_temp_storage.Allocate(1, target));
      GUARD_CU(visited.Allocate(sub_graph.nodes, target));
      GUARD_CU(neighbour_labels.Allocate(sub_graph.edges+1, target));
      
      GUARD_CU(split_lengths.Allocate(2, util::HOST | target));

      if (num_gpus > 1) {

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

      been_in_backward = false;

      GUARD_CU(util::SetDevice(this->gpu_idx));

      // Allocate output labels if necessary
      GUARD_CU(labels.EnsureSize_(nodes, target));
      GUARD_CU(visited.EnsureSize_(nodes, target));

      GUARD_CU(visited.ForEach( [] __host__ __device__(int &x) { x = (int)0; }, nodes, target, this->stream));
      GUARD_CU(neighbour_labels_size.ForEach( [] __host__ __device__(int &x) { x = (int)0; }, nodes, target, this->stream));
      GUARD_CU(segments_size.ForEach( [] __host__ __device__(int &x) { x = (int)0; }, nodes, target, this->stream));

      return retval;
    }  // end of Reset
  };   // end of DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief LPProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief LPProblem default destructor
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
   * @brief Copy result labels computed on the GPU back to
   *host-side vectors.
   * @param[out] h_labels Host array to store computed vertex labels
   * @param[in]  target where the results are stored
   * \return     cudaError_t Error message(s), if any
   */
  cudaError_t Extract(LabelT *h_labels,
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

      }

      else if (target == util::HOST) {
        GUARD_CU(data_slice.labels.ForAll(
            [h_labels] __host__ __device__(const LabelT *labels,
                                           const VertexT &v) {
              h_labels[v] = labels[v];
            },
            nodes, util::HOST));
      }
    }

    else {  // num_gpus != 1
      util::Array1D<SizeT, LabelT *> th_labels;
      th_labels.SetName("lp::Problem::Extract::th_labels");
      GUARD_CU(th_labels.Allocate(this->num_gpus, util::HOST));

      for (int gpu = 0; gpu < this->num_gpus; gpu++) {
        auto &data_slice = data_slices[gpu][0];
        if (target == util::DEVICE) {
          GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
          GUARD_CU(data_slice.labels.Move(util::DEVICE, util::HOST));
        }
        th_labels[gpu] = data_slice.labels.GetPointer(util::HOST);
      }  // end for(gpu)

      for (VertexT v = 0; v < nodes; v++) {
        int gpu = this->org_graph->GpT::partition_table[v];
        VertexT v_ = v;
        if ((GraphT::FLAG & gunrock::partitioner::Keep_Node_Num) == 0)
          v_ = this->org_graph->GpT::convertion_table[v];

        h_labels[v] = th_labels[gpu][v_];
      }

      GUARD_CU(th_labels.Release());
    }  // end if (num_gpus ==1)

    return retval;
  }

  /**
   * @brief initialization function.
   * @param     graph       The graph that LP processes on
   * @param[in] Location    Memory location to work on
   * \return    cudaError_t Error message(s), if any
   */
  cudaError_t Init(GraphT &graph, util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    GUARD_CU(BaseProblem::Init(graph, target));
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

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

    // Fillin the initial input_queue
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
    }
    printf("This is where the util:DEVICE=2 is checked with target %d\n", target);

    if (target & util::DEVICE) {
      GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

      GUARD_CU(data_slices[gpu]->labels.SetIdx());

      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    }

    return retval;
  }  // end of reset

  /** @} */

};  // end of problem

}  // namespace lp 
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
