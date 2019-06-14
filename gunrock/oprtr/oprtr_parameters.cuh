// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * oprtr_parameters.cuh
 *
 * @brief Data structure for operator parameters
 */

#pragma once

#include <moderngpu.cuh>
#include <gunrock/app/frontier.cuh>
#include <gunrock/oprtr/oprtr_base.cuh>

namespace gunrock {
namespace oprtr {

template <typename GraphT, typename FrontierT, typename _LabelT>
struct OprtrParameters {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef _LabelT LabelT;

  // app::EnactorStats     <SizeT> *enactor_stats;
  FrontierT *frontier;
  util::Array1D<SizeT, ValueT> *values_in;
  util::Array1D<SizeT, ValueT> *values_out;
  util::Array1D<SizeT, ValueT> *reduce_values_temp;
  util::Array1D<SizeT, ValueT> *reduce_values_temp2;
  util::Array1D<SizeT, ValueT> *reduce_values_out;
  util::Array1D<SizeT, SizeT> *vertex_markers;
  util::Array1D<SizeT, unsigned char> *visited_masks;
  util::Array1D<SizeT, LabelT> *labels;
  util::CudaProperties *cuda_props;

  // VertexT *d_backward_index_queue;
  // bool    *d_backward_frontier_map_in;
  // bool    *d_backward_frontier_map_out;
  // SizeT         max_in;
  // SizeT         max_out;
  mgpu::ContextPtr context;
  cudaStream_t stream;
  bool get_output_length;
  bool reduce_reset;
  std::string advance_mode;
  std::string filter_mode;
  // bool          filtering_flag;
  // bool          skip_marking;
  LabelT label;
  int max_grid_size;

  OprtrParameters() { Init(); }

  ~OprtrParameters() { Init(); }

  cudaError_t Init() {
    // enactor_stats      = NULL;
    frontier = NULL;
    values_in = NULL;
    values_out = NULL;
    // output_offsets     = NULL;
    reduce_values_temp = NULL;
    reduce_values_temp2 = NULL;
    reduce_values_out = NULL;
    vertex_markers = NULL;
    visited_masks = NULL;
    labels = NULL;
    cuda_props = NULL;
    // max_in             = 0;
    // max_out            = 0;
    // context            = NULL;
    stream = 0;
    get_output_length = true;
    reduce_reset = true;
    advance_mode = "";
    filter_mode = "";
    max_grid_size = 0;
    return cudaSuccess;
  }
};

}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
