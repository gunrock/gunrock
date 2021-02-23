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
#ifdef __CUDA_ARCH__
              true
#else
              false
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
      const SizeT offset,
      const InKeyT *&d_src_node_ids, const VertexT *&d_dst_node_ids,
      SizeT &input_length,
      InterOpT &inter_op) {
    VertexT start = threadIdx.x + blockIdx.x * blockDim.x;
    for (VertexT idx = start; idx < input_length;
         idx += KernelPolicyT::BLOCKS * KernelPolicyT::THREADS) {
      // get nls start and end index for two ids
      VertexT sid = _ldg(d_src_node_ids + idx);
      VertexT did = _ldg(d_dst_node_ids + idx + offset);
      if (sid >= did) continue;
      SizeT src_it = _ldg(d_row_offsets + sid);
      SizeT src_end = _ldg(d_row_offsets + sid + 1);
      SizeT dst_it = _ldg(d_row_offsets + did);
      SizeT dst_end = _ldg(d_row_offsets + did + 1);
      if (src_it >= src_end || dst_it >= dst_end) continue;
      SizeT src_nl_size = src_end - src_it;
      SizeT dst_nl_size = dst_end - dst_it;
      SizeT min_nl = (src_nl_size > dst_nl_size) ? dst_nl_size : src_nl_size;
      SizeT max_nl = (src_nl_size < dst_nl_size) ? dst_nl_size : src_nl_size;
      if (min_nl * ilog2((unsigned int)(max_nl)) * 10 < min_nl + max_nl) {
        // search
        SizeT min_it = (src_nl_size < dst_nl_size) ? src_it : dst_it;
        SizeT min_end = min_it + min_nl;
        SizeT max_it = (src_nl_size < dst_nl_size) ? dst_it : src_it;
        VertexT *keys = &d_column_indices[max_it + offset];
        while (min_it < min_end) {
          VertexT small_edge = d_column_indices[(min_it++) + offset];
          if (BinarySearch(keys, max_nl, small_edge) == 1) {
            inter_op(small_edge, idx);
          }
        }
      } else {
        VertexT src_edge = _ldg(d_column_indices + src_it);
        VertexT dst_edge = _ldg(d_column_indices + dst_it);
        while (src_it < src_end && dst_it < dst_end) {
          int diff = src_edge - dst_edge;
          if (diff == 0) {
            inter_op(src_edge, idx);
          }
          src_edge =
              (diff <= 0) ? _ldg(d_column_indices + (++src_it)) : src_edge;
          dst_edge =
              (diff >= 0) ? _ldg(d_column_indices + (++dst_it)) : dst_edge;
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
                             const SizeT offset, 
                             const InKeyT *d_src_node_ids,
                             const VertexT *d_dst_node_ids,
 
                             SizeT input_length,
                             InterOpT inter_op) {
  Dispatch<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT,
           InterOpT>::IntersectTwoSmallNL(d_row_offsets, d_column_indices,
                                          offset,
                                          d_src_node_ids, d_dst_node_ids, 
                                          input_length,
                                          inter_op);
}

template <OprtrFlag FLAG, typename GraphT, typename FrontierInT,
          typename FrontierOutT, typename ParametersT, typename InterOpT>
cudaError_t Launch(gunrock::util::MultiGpuContext mgpu_context,
    const GraphT &graph, const FrontierInT *frontier_in,
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

  util::SaveToRestore state;
  state.Save();

  size_t input_length = graph.edges;
  SizeT num_vertices = graph.nodes;

  printf("Num verts %d\n", num_vertices);

  std::vector<std::thread> threads;
  threads.reserve(mgpu_context.getGpuCount());

  for (auto& context : mgpu_context.contexts) {

    threads.push_back(std::thread( [&, context]() {
        SizeT edges_per_gpu = gunrock::util::ceil_divide(input_length, mgpu_context.getGpuCount());

        auto offset = edges_per_gpu * context.device_id;

        auto rows_offset = graph.CsrT::row_offsets.GetPointer(util::DEVICE);
        auto column_indices = graph.CsrT::column_indices.GetPointer(util::DEVICE);

        printf("offset %p\n", offset);
        printf("row offsets %p\n", rows_offset);
        printf("col indicies %p\n", column_indices);

        cudaSetDevice(context.device_id);
        IntersectTwoSmallNL<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT, InterOpT>
        <<<KernelPolicyT::BLOCKS, KernelPolicyT::THREADS, 0, context.stream>>>(
            rows_offset,
            column_indices,
            offset,
            frontier_in->GetPointer(util::DEVICE),
            column_indices,
            edges_per_gpu,
            inter_op);
        cudaEventRecord(context.event, context.stream);
    }));
  }

  for (auto &thread : threads) {
    thread.join();
  }

  for (auto & context : mgpu_context.contexts) {
    cudaStreamWaitEvent(parameters.stream, context.event, 0);
  }

  state.Restore();

  // IntersectTwoSmallNL<FLAG, InKeyT, OutKeyT, SizeT, ValueT, VertexT, InterOpT>
  //     <<<KernelPolicyT::BLOCKS, KernelPolicyT::THREADS, 0, parameters.stream>>>(
  //         graph.CsrT::row_offsets.GetPointer(util::DEVICE),
  //         graph.CsrT::column_indices.GetPointer(util::DEVICE),
  //         frontier_in->GetPointer(util::DEVICE),
  //         graph.CsrT::column_indices.GetPointer(util::DEVICE),
  //         frontier_out->GetPointer(util::DEVICE), input_length, stride,
  //         num_vertices, inter_op);

  return cudaSuccess;
}

}  // namespace intersection
}  // namespace oprtr
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
