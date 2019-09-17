// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sage_problem.cuh
 *
 * @brief GPU Storage management Structure for SSSP Problem Data
 */

#pragma once

#include <gunrock/app/problem_base.cuh>

namespace gunrock {
namespace app {
namespace sage {

/**
 * @brief Speciflying parameters for SSSP Problem
 * @param  parameters  The util::Parameter<...> structure holding all parameter
 * info \return cudaError_t error message(s), if any
 */
cudaError_t UseParameters_problem(util::Parameters &parameters) {
  cudaError_t retval = cudaSuccess;

  GUARD_CU(gunrock::app::UseParameters_problem(parameters));
  // GUARD_CU(parameters.Use<bool>(
  //    "mark-pred",
  //    util::OPTIONAL_ARGUMENT | util::MULTI_VALUE | util::OPTIONAL_PARAMETER,
  //    false,
  //    "Whether to mark predecessor info.",
  //    __FILE__, __LINE__));

  return retval;
}

/**
 * @brief Single-Source Shortest Path Problem structure.
 * @tparam _GraphT  Type of the graph
 * @tparam _LabelT  Type of labels used in sage
 * @tparam _ValueT  Type of per-vertex distance values
 * @tparam _FLAG    Problem flags
 */
template <typename _GraphT,
          // typename _LabelT = typename _GraphT::VertexT,
          typename _ValueT = typename _GraphT::ValueT,
          ProblemFlag _FLAG = Problem_None>
struct Problem : ProblemBase<_GraphT, _FLAG> {
  typedef _GraphT GraphT;
  static const ProblemFlag FLAG = _FLAG;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  //  typedef typename GraphT::CsrT    CsrT;
  typedef typename GraphT::GpT GpT;
  typedef typename GraphT::VertexT LabelT;
  typedef _ValueT ValueT;

  typedef ProblemBase<GraphT, FLAG> BaseProblem;
  typedef DataSliceBase<GraphT, FLAG> BaseDataSlice;

  // Helper structures

  /**
   * @brief Data structure containing SSSP-specific data on indivual GPU.
   */
  struct DataSlice : BaseDataSlice {
    // sage-specific storage arrays
    util::Array1D<SizeT, ValueT>
        W_f_1_1D;  // w_f_1 1D array. weight matrix for W^1 feature part
    util::Array1D<SizeT, ValueT>
        W_a_1_1D;  // w_a_1 1D array. weight matrix for W^1 agg part
    util::Array1D<SizeT, ValueT>
        W_f_2_1D;  // w_f_2 1D array. weight matrix for W^2 feature part
    util::Array1D<SizeT, ValueT>
        W_a_2_1D;  // w_a_2 1D array. weight matrix for W^2 agg part
    util::Array1D<uint64_t, ValueT> features_1D;  // fature matrix 1D
    util::Array1D<SizeT, ValueT> children_temp;   // 256 agg(h_B1^1)
    util::Array1D<SizeT, ValueT> source_temp;     // 256 h_B2^1
    util::Array1D<SizeT, ValueT> source_result;   // 256 h_B2^2
    util::Array1D<SizeT, ValueT>
        child_temp;  // 256 h_B1^1, I feel like this one could be local
    util::Array1D<SizeT, ValueT>
        sums_child_feat;  // 64 sum of children's features, I feel like this one
                          // could be local as well
    util::Array1D<SizeT, ValueT> sums;  // 64 per child
    util::Array1D<uint64_t, ValueT, util::PINNED>
        host_source_result;  // results on HOST

    util::Array1D<SizeT, curandState>
        rand_states;  // random states, one per child

    util::Array1D<SizeT, VertexT> children;  // children vertices

    VertexT batch_size;
    int feature_column;
    int num_children_per_source;
    int num_leafs_per_child;
    int Wf1_dim0, Wf1_dim1;
    int Wa1_dim0, Wa1_dim1;
    int Wf2_dim0, Wf2_dim1;
    int Wa2_dim0, Wa2_dim1;
    int result_column;
    bool custom_kernels;
    bool debug;
    cudaStream_t d2h_stream;
    cudaEvent_t d2h_start, d2h_finish;

    /*
     * @brief Default constructor
     */
    DataSlice() : BaseDataSlice() {
      W_f_1_1D.SetName("W_f_1_1D");
      W_a_1_1D.SetName("W_a_1_1D");
      W_f_2_1D.SetName("W_f_2_1D");
      W_a_2_1D.SetName("W_a_2_1D");
      features_1D.SetName("features_1D");
      children_temp.SetName("children_temp");
      source_temp.SetName("source_temp");
      source_result.SetName("source_result");
      child_temp.SetName("child_temp");
      sums_child_feat.SetName("sums_child_feat");
      sums.SetName("sums");
      host_source_result.SetName("host_source_result");
      rand_states.SetName("rand_states");
      children.SetName("children");
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

      GUARD_CU(W_f_1_1D.Release(target));
      GUARD_CU(W_a_1_1D.Release(target));
      GUARD_CU(W_f_2_1D.Release(target));
      GUARD_CU(W_a_2_1D.Release(target));
      GUARD_CU(features_1D.Release(target));
      GUARD_CU(children_temp.Release(target));
      GUARD_CU(source_temp.Release(target));
      GUARD_CU(source_result.Release(target));
      GUARD_CU(child_temp.Release(target));
      GUARD_CU(sums_child_feat.Release(target));
      GUARD_CU(sums.Release(target));
      GUARD_CU(host_source_result.Release(util::HOST));
      GUARD_CU(rand_states.Release(target));
      GUARD_CU(children.Release(target));
      GUARD_CU2(cudaStreamDestroy(d2h_stream), "cudaStreamDestory failed.");
      GUARD_CU2(cudaEventDestroy(d2h_start), "cudaEventDestory failed.");
      GUARD_CU2(cudaEventDestroy(d2h_finish), "cudaEventDestory failed.");
      GUARD_CU(BaseDataSlice ::Release(target));

      return retval;
    }

    /**
     * @brief initializing sage-specific data on each gpu
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
      auto nodes = sub_graph.nodes;

      GUARD_CU(BaseDataSlice::Init(sub_graph, num_gpus, gpu_idx, target, flag));
      GUARD_CU(W_f_1_1D.Allocate(  // 64 *128, target));
          Wf1_dim0 * Wf1_dim1, target));
      GUARD_CU(W_a_1_1D.Allocate(  // 64 *128, target));
          Wa1_dim0 * Wa1_dim1, target));
      GUARD_CU(W_f_2_1D.Allocate(  // 256*128, target));
          Wf2_dim0 * Wf2_dim1, target));
      GUARD_CU(W_a_2_1D.Allocate(  // 256*128, target));
          Wa2_dim0 * Wa2_dim1, target));
      GUARD_CU(
          features_1D.Allocate(((uint64_t)nodes) * feature_column, target));

      auto num_children = num_children_per_source * batch_size;
      if (!custom_kernels || Wa2_dim0 > 1024) {
        GUARD_CU(child_temp.Allocate(num_children * Wf2_dim0, target));
      }
      GUARD_CU(children_temp.Allocate(batch_size * Wf2_dim0, target));
      GUARD_CU(source_temp.Allocate(batch_size * Wf2_dim0, target));
      GUARD_CU(sums_child_feat.Allocate(batch_size * result_column, target));
      GUARD_CU(sums.Allocate(num_children * feature_column, target));
      GUARD_CU(source_result.Allocate(batch_size * result_column, target));
      GUARD_CU(rand_states.Allocate(
          max(80 * 256, 2560 * min(feature_column, 512)), target));
      GUARD_CU(children.Allocate(num_children, target));

      GUARD_CU(host_source_result.Allocate(((uint64_t)nodes) * result_column,
                                           util::HOST));
      GUARD_CU2(cudaStreamCreateWithFlags(&d2h_stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags failed.");
      GUARD_CU2(cudaEventCreateWithFlags(&d2h_start, cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed.");
      GUARD_CU2(cudaEventCreateWithFlags(&d2h_finish, cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed.");

      GUARD_CU(sub_graph.Move(util::HOST, target, this->stream));
      return retval;
    }  // Init

    /**
     * @brief Reset problem function. Must be called prior to each run.
     * @param[in] target      Targeting device location
     * \return    cudaError_t Error message(s), if any
     */
    cudaError_t Reset(util::Location target = util::DEVICE) {
      cudaError_t retval = cudaSuccess;

      // Reset data
      GUARD_CU(child_temp.ForEach(
          [] __host__ __device__(ValueT & val) { val = 0; },
          util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));

      GUARD_CU(children_temp.ForEach(
          [] __host__ __device__(ValueT & val) { val = 0; },
          util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));

      GUARD_CU(source_temp.ForEach(
          [] __host__ __device__(ValueT & val) { val = 0; },
          util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));

      GUARD_CU(sums_child_feat.ForEach(
          [] __host__ __device__(ValueT & val) { val = 0; },
          util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));

      GUARD_CU(sums.ForEach([] __host__ __device__(ValueT & val) { val = 0; },
                            util::PreDefinedValues<SizeT>::InvalidValue, target,
                            this->stream));

      GUARD_CU(source_result.ForEach(
          [] __host__ __device__(ValueT & val) { val = 0; },
          util::PreDefinedValues<SizeT>::InvalidValue, target, this->stream));

      return retval;
    }
  };  // DataSlice

  // Members
  // Set of data slices (one for each GPU)
  util::Array1D<SizeT, DataSlice> *data_slices;

  // Methods

  /**
   * @brief SSSPProblem default constructor
   */
  Problem(util::Parameters &_parameters, ProblemFlag _flag = Problem_None)
      : BaseProblem(_parameters, _flag), data_slices(NULL) {}

  /**
   * @brief SSSPProblem default destructor
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
  cudaError_t Extract(ValueT *h_source_result,
                      // VertexT        *h_preds     = NULL,
                      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;
    SizeT nodes = this->org_graph->nodes;
    auto &data_slice = data_slices[0][0];

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");
    GUARD_CU(data_slice.host_source_result.ForEach(
        h_source_result,
        [] __host__ __device__(const ValueT &val, ValueT &h_val) {
          h_val = val;
        },
        ((uint64_t)nodes) * data_slice.result_column, util::HOST));

    return retval;
  }

  template <typename ArrayT>
  cudaError_t ReadMat(ArrayT &array, std::string filename, uint64_t dim0,
                      uint64_t dim1) {
    cudaError_t retval = cudaSuccess;

    auto temp_vals_2D = gunrock::app::sage::template ReadMatrix<ValueT, SizeT>(
        filename, dim0, dim1);
    GUARD_CU(array.Allocate(dim0 * dim1, util::HOST));
    // for (auto pos = 0; pos < dim0 * dim1; pos++)
    //{
    //    array[pos] = temp_vals_2D[pos / dim1][pos % dim1];
    //}
    GUARD_CU(array.ForAll(
        [temp_vals_2D, dim1] __host__ __device__(ValueT * vals,
                                                 const uint64_t &pos) {
          vals[pos] = temp_vals_2D[pos / dim1][pos % dim1];
        },
        dim0 * dim1, util::HOST));
    for (auto x = 0; x < dim0; x++) {
      delete[] temp_vals_2D[x];
      temp_vals_2D[x] = NULL;
    }
    delete[] temp_vals_2D;
    temp_vals_2D = NULL;

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
    auto &para = this->parameters;
    data_slices = new util::Array1D<SizeT, DataSlice>[this->num_gpus];

    //    if (this -> parameters.template Get<bool>("mark-pred"))
    //        this -> flag = this -> flag | Mark_Predecessors;

    for (int gpu = 0; gpu < this->num_gpus; gpu++) {
      data_slices[gpu].SetName("data_slices[" + std::to_string(gpu) + "]");
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));

      GUARD_CU(data_slices[gpu].Allocate(1, target | util::HOST));

      auto &data_slice = data_slices[gpu][0];
      data_slice.batch_size = para.template Get<int>("batch-size");
      data_slice.feature_column = para.template Get<int>("feature-column");
      data_slice.num_children_per_source =
          para.template Get<int>("num-children-per-source");
      data_slice.Wf1_dim0 = data_slice.feature_column;
      data_slice.Wf1_dim1 = para.template Get<int>("Wf1-dim1");
      data_slice.Wa1_dim0 = data_slice.feature_column;
      data_slice.Wa1_dim1 = para.template Get<int>("Wa1-dim1");
      data_slice.Wf2_dim0 = data_slice.Wf1_dim1 + data_slice.Wa1_dim1;
      data_slice.Wf2_dim1 = para.template Get<int>("Wf2-dim1");
      data_slice.Wa2_dim0 = data_slice.Wf1_dim1 + data_slice.Wa1_dim1;
      data_slice.Wa2_dim1 = para.template Get<int>("Wa2-dim1");
      data_slice.result_column = data_slice.Wa2_dim1 + data_slice.Wf2_dim1;
      data_slice.num_leafs_per_child =
          para.template Get<int>("num-leafs-per-child");
      if (!util::isValid(data_slice.num_leafs_per_child))
        data_slice.num_leafs_per_child = data_slice.num_children_per_source;
      data_slice.custom_kernels = para.template Get<bool>("custom-kernels");
      data_slice.debug = para.template Get<bool>("v");

      GUARD_CU(data_slice.Init(this->sub_graphs[gpu], this->num_gpus,
                               this->gpu_idx[gpu], target, this->flag));

      std::string Wf1_filename = para.template Get<std::string>("Wf1");
      if (Wf1_filename == "") util::PrintMsg("Using randomly generated Wf1");
      GUARD_CU(ReadMat(data_slice.W_f_1_1D, Wf1_filename, data_slice.Wf1_dim0,
                       data_slice.Wf1_dim1));

      std::string Wa1_filename = para.template Get<std::string>("Wa1");
      if (Wa1_filename == "") util::PrintMsg("Using randomly generated Wa1");
      GUARD_CU(ReadMat(data_slice.W_a_1_1D, Wa1_filename, data_slice.Wa1_dim0,
                       data_slice.Wa1_dim1));

      std::string Wf2_filename = para.template Get<std::string>("Wf2");
      if (Wf2_filename == "") util::PrintMsg("Using randomly generated Wf2");
      GUARD_CU(ReadMat(data_slice.W_f_2_1D, Wf2_filename, data_slice.Wf2_dim0,
                       data_slice.Wf2_dim1));

      std::string Wa2_filename = para.template Get<std::string>("Wa2");
      if (Wa2_filename == "") util::PrintMsg("Using randomly generated Wa2");
      GUARD_CU(ReadMat(data_slice.W_a_2_1D, Wa2_filename, data_slice.Wa2_dim0,
                       data_slice.Wa2_dim1));

      std::string features_filename =
          para.template Get<std::string>("features");
      if (features_filename == "")
        util::PrintMsg("Using randomly generated features");
      GUARD_CU(ReadMat(data_slice.features_1D, features_filename, graph.nodes,
                       data_slice.feature_column));

      GUARD_CU(data_slice.W_f_1_1D.Move(util::HOST, util::DEVICE));
      GUARD_CU(data_slice.W_a_1_1D.Move(util::HOST, util::DEVICE));
      GUARD_CU(data_slice.W_f_2_1D.Move(util::HOST, util::DEVICE));
      GUARD_CU(data_slice.W_a_2_1D.Move(util::HOST, util::DEVICE));
      GUARD_CU(data_slice.features_1D.Move(util::HOST, util::DEVICE));
      GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed.");
    }  // end for (gpu)

    return retval;
  }

  /**
   * @brief Reset problem function. Must be called prior to each run.
   * @param[in] src      Source vertex to start.
   * @param[in] location Memory location to work on
   * \return cudaError_t Error message(s), if any
   */
  cudaError_t Reset(
      // VertexT    src,
      util::Location target = util::DEVICE) {
    cudaError_t retval = cudaSuccess;

    for (int gpu = 0; gpu < this->num_gpus; ++gpu) {
      // Set device
      if (target & util::DEVICE) GUARD_CU(util::SetDevice(this->gpu_idx[gpu]));
      GUARD_CU(data_slices[gpu]->Reset(target));

      int rand_seed = this->parameters.template Get<int>("rand-seed");
      if (!util::isValid(rand_seed)) rand_seed = time(NULL);
      if (!this->parameters.template Get<bool>("quiet"))
        util::PrintMsg("rand-seed = " + std::to_string(rand_seed));
      GUARD_CU(data_slices[gpu]->rand_states.ForAll(
          [rand_seed] __host__ __device__(curandState * states,
                                          const SizeT &pos) {
            curand_init(rand_seed, pos, 0, states + pos);
          },
          util::PreDefinedValues<SizeT>::InvalidValue, util::DEVICE,
          data_slices[gpu]->stream));

      GUARD_CU(data_slices[gpu].Move(util::HOST, target));
    }

    GUARD_CU2(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed");

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
