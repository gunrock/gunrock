// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * kernel.cuh
 *
 * @brief Intersection Kernel Entry point
 *
 * Expected inputs are two arrays of node IDs, each pair of nodes forms an edge.
 * The intersections of each node pair's neighbor lists are computed and
 * returned as a single usnigned int value. Can perform user-defined functors on
 * each of these intersection.
 */

#pragma once
#include <gunrock/util/cta_work_distribution.cuh>
#include <gunrock/util/cta_work_progress.cuh>
#include <gunrock/util/kernel_runtime_stats.cuh>

#include <gunrock/oprtr/intersection/cta.cuh>

#include <gunrock/oprtr/intersection/kernel_policy.cuh>
#include <cub/cub.cuh>
#include <moderngpu.cuh>
#include <gunrock/util/test_utils.cuh>

namespace gunrock {
namespace oprtr {
namespace intersection {

/**
 * Arch dispatch
 */

/**
 * Not valid for this arch (default)
 * @tparam FLAG Operator flags
 * @tparam GraphT Graph type
 * @tparam InKeyT Input keys type
 * @tparam OutKeyT Output keys type
 * @tparam VALID
 */
template <OprtrFlag FLAG, typename InKeyT, typename OutKeyT, typename SizeT,
          typename ValueT, typename VertexT, typename InterOpt,
          bool VALID =
#ifndef __CUDA_ARCH__
              false
#else
              (__CUDA_ARCH__ >= CUDA_ARCH)
#endif
          >
struct Dispatch {
};

template <OprtrFlag FLAG, typename InKeyT, typename OutKeyT, typename SizeT,
          typename ValueT, typename VertexT, typename InterOpT>
struct Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT, InterOpT, true> {
  typedef KernelPolicy<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT, InterOpT,
                       1, 10, 8, 30>
      KernelPolicyT;

  static __device__ void IntersectTwoSmallNL(
      const SizeT *&d_row_offsets, VertexT *&d_column_indices,
      const InKeyT *&d_src_node_ids, const VertexT *&d_dst_node_ids,
      //       ValueT        *&d_output_counts,
      OutKeyT *&d_output_total, SizeT &input_length, SizeT &stride,
      SizeT &num_vertex, SizeT &num_edge, InterOpT &inter_op) {
    VertexT start = threadIdx.x + blockIdx.x * blockDim.x;
    for (VertexT idx = start; idx < input_length;
         idx += KernelPolicyT::BLOCKS * KernelPolicyT::THREADS) {
      // get nls start and end index for two ids
      VertexT sid = __ldg(d_src_node_ids + idx);
      VertexT did = __ldg(d_dst_node_ids + idx);
      if (sid >= did) continue;
      SizeT src_it = __ldg(d_row_offsets + sid);
      SizeT src_end = __ldg(d_row_offsets + sid + 1);
      SizeT dst_it = __ldg(d_row_offsets + did);
      SizeT dst_end = __ldg(d_row_offsets + did + 1);
      if (src_it >= src_end || dst_it >= dst_end) continue;
      SizeT src_nl_size = src_end - src_it;
      SizeT dst_nl_size = dst_end - dst_it;
      SizeT min_nl = (src_nl_size > dst_nl_size) ? dst_nl_size : src_nl_size;
      SizeT max_nl = (src_nl_size < dst_nl_size) ? dst_nl_size : src_nl_size;
      SizeT total = min_nl + max_nl;
      if (min_nl * ilog2((unsigned int)(max_nl)) * 10 < min_nl + max_nl) {
        // search
        SizeT min_it = (src_nl_size < dst_nl_size) ? src_it : dst_it;
        SizeT min_end = min_it + min_nl;
        SizeT max_it = (src_nl_size < dst_nl_size) ? dst_it : src_it;
        VertexT *keys = &d_column_indices[max_it];
        while (min_it < min_end) {
          VertexT small_edge = d_column_indices[min_it++];
          if (BinarySearch(keys, max_nl, small_edge) == 1) {
            inter_op(small_edge, idx);
          }
        }
      } else {
        VertexT src_edge = __ldg(d_column_indices + src_it);
        VertexT dst_edge = __ldg(d_column_indices + dst_it);
        while (src_it < src_end && dst_it < dst_end) {
          int diff = src_edge - dst_edge;
          if (diff == 0) {
            inter_op(src_edge, idx);
          }
          src_edge =
              (diff <= 0) ? __ldg(d_column_indices + (++src_it)) : src_edge;
          dst_edge =
              (diff >= 0) ? __ldg(d_column_indices + (++dst_it)) : dst_edge;
        }
      }
    }

  }  // IntersectTwoSmallNL
};

/**
 * @brief Kernel entry for IntersectTwoSmallNL function
 *
 * @tparam KernelPolicy Kernel policy type for intersection.
 * @tparam ProblemData Problem data type for intersection.
 * @tparam Functor Functor type for the specific problem type.
 *
 * @param[in] d_row_offset      Device pointer of SizeT to the row offsets queue
 * @param[in] d_column_indices  Device pointer of VertexT to the column indices
 * queue
 * @param[in] d_src_node_ids    Device pointer of VertexT to the incoming
 * frontier queue (source node ids)
 * @param[in] d_dst_node_ids    Device pointer of VertexT to the incoming
 * frontier queue (destination node ids)
 * @param[in] d_edge_list       Device pointer of VertexT to the edge list IDs
 * @param[in] d_degrees         Device pointer of SizeT to degree array
 * @param[in] problem           Device pointer to the problem object
 * @param[out] d_output_counts  Device pointer to the output counts array
 * @param[in] input_length      Length of the incoming frontier queues
 * (d_src_node_ids and d_dst_node_ids should have the same length)
 * @param[in] num_vertex        Maximum number of elements we can place into the
 * incoming frontier
 * @param[in] num_edge          Maximum number of elements we can place into the
 * outgoing frontier
 *
 */
template <OprtrFlag FLAG, typename InKeyT, typename OutKeyT, typename SizeT,
          typename ValueT, typename VertexT, typename InterOpT>
__launch_bounds__(Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT,
                           InterOpT, true>::KernelPolicyT::THREADS,
                  Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT,
                           InterOpT, true>::KernelPolicyT::CTA_OCCUPANCY)
    __global__
    void IntersectTwoSmallNL(const SizeT *d_row_offsets,
                             VertexT *d_column_indices,
                             const InKeyT *d_src_node_ids,
                             const VertexT *d_dst_node_ids,
                             //      	  ValueT       *d_output_counts,
                             OutKeyT *d_output_total, SizeT input_length,
                             SizeT stride, SizeT num_vertex, SizeT num_edge,
                             InterOpT inter_op) {
  Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT,
           InterOpT>::IntersectTwoSmallNL(d_row_offsets, d_column_indices,
                                          d_src_node_ids, d_dst_node_ids,
                                          //      d_output_counts,
                                          d_output_total, input_length, stride,
                                          num_vertex, num_edge, inter_op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename InterOpT>
cudaError_t Launch(const GraphT &graph, const FrontierInT *frontier_in,
                   FrontierOutT *frontier_out, ParametersT &parameters,
                   InterOpT inter_op) {
  typedef typename GraphT ::CsrT CsrT;
  typedef typename FrontierInT ::ValueT InKeyT;
  typedef typename FrontierOutT::ValueT OutKeyT;
  typedef typename ParametersT ::SizeT SizeT;
  typedef typename ParametersT ::ValueT ValueT;
  typedef typename ParametersT ::VertexT VertexT;
  typedef typename ParametersT ::LabelT LabelT;
  typedef typename Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT,
                            InterOpT, true>::KernelPolicyT KernelPolicyT;

  size_t input_length = graph.edges;
  size_t stride =
      (input_length + KernelPolicyT::BLOCKS * KernelPolicyT::THREADS - 1) >>
      (KernelPolicyT::LOG_THREADS + KernelPolicyT::LOG_BLOCKS);
  SizeT num_vertex = graph.nodes;
  SizeT num_edges = graph.edges;

  IntersectTwoSmallNL<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT, InterOpT>
      <<<KernelPolicyT::BLOCKS, KernelPolicyT::THREADS, 0, parameters.stream>>>(
          graph.CsrT::row_offsets.GetPointer(util::DEVICE),
          graph.CsrT::column_indices.GetPointer(util::DEVICE),
          frontier_in->GetPointer(util::DEVICE),
          graph.CsrT::column_indices.GetPointer(util::DEVICE),
          frontier_out->GetPointer(util::DEVICE), input_length, stride,
          num_vertex, num_edges, inter_op);

  return cudaSuccess;  //(float)tc_count / (float)total;
}

}  // namespace intersection
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
