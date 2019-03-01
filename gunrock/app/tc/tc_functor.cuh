#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/tc/tc_problem.cuh>

namespace gunrock {
namespace app {
namespace tc {

template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct TCFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ bool CondEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos) {
    // bool res =  (problem->d_degrees[s_id] > problem->d_degrees[d_id]
    //        || (problem->d_degrees[s_id] == problem->d_degrees[d_id] && s_id <
    //        d_id));
    bool res = s_id < d_id;
    d_data_slice->d_src_node_ids[edge_id] = (res) ? 1 : 0;
    return res;
  }

  static __device__ __forceinline__ void ApplyEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos) {
    return;
  }

  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //    printf("%d\n",node);
    return (node != -1);
  }

  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    return;
  }
};

}  // namespace tc
}  // namespace app
}  // namespace gunrock
