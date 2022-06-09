// ----------------------------------------------------------------------------
// Gunrock -- High-Performance Graph Primitives on GPU
// ----------------------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------------------

/**
 * @file lp_functor.cuh
 * @brief Device functions
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/lp/lp_problem.cuh>
#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace app {
namespace lp {

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Structure contains device functions in label propagation.
 *
 * @tparam VertexId    Type of signed integer use as vertex identifier
 * @tparam SizeT       Type of integer / unsigned integer for array indexing
 * @tparam Value       Type of integer / float / double to attributes
 * @tparam ProblemData Problem data type contains data slice for MST problem
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct LpHookFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ bool CondFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    VertexId from = _ldg(d_data_slice->froms + node);
    VertexId to = _ldg(d_data_slice->tos + node);
    Value weight_from = _ldg(d_data_slice->node_weights + from);
    Value weight_to = _ldg(d_data_slice->node_weights + to);
    return fabs(weight_from - weight_to) < 1e-5f;
  }

  /**
   * @brief Filter Kernel apply function. Point the current node to the
   * parent node of its parent node.
   *
   * @param[in] node Vertex Id
   * @param[in] problem Data slice object
   * @param[in] v Vertex value
   * @param[in] nid Node ID
   */
  static __device__ __forceinline__ void ApplyFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    VertexId from = _ldg(d_data_slice->froms + node);
    VertexId to = _ldg(d_data_slice->tos + node);
    VertexId max_node = from > to ? from : to;
    VertexId min_node = from > to ? to : from;
    d_data_slice->labels[max_node] = min_node;
  }
};

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Structure contains device functions in label propagation.
 *
 * @tparam VertexId    Type of signed integer use as vertex identifier
 * @tparam SizeT       Type of integer / unsigned integer for array indexing
 * @tparam Value       Type of integer / float / double to attributes
 * @tparam ProblemData Problem data type contains data slice for MST problem
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct LpPtrJumpFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ bool CondFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    return true;
  }

  /**
   * @brief Filter Kernel apply function. Point the current node to the
   * parent node of its parent node.
   *
   * @param[in] node Vertex Id
   * @param[in] problem Data slice object
   * @param[in] v Vertex value
   * @param[in] nid Node ID
   */
  static __device__ __forceinline__ void ApplyFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    VertexId parent = d_data_slice->labels[node];
    VertexId grandpa = d_data_slice->labels[parent];
    if (parent != grandpa) {
      d_data_slice->stable_flag[0] = 0;
      d_data_slice->labels[node] = grandpa;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Structure contains device functions in label propagation.
 *
 * @tparam VertexId    Type of signed integer use as vertex identifier
 * @tparam SizeT       Type of integer / unsigned integer for array indexing
 * @tparam Value       Type of integer / float / double to attributes
 * @tparam ProblemData Problem data type contains data slice for MST problem
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct LpWeightUpdateFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ bool CondFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    return true;
  }

  /**
   * @brief Filter Kernel apply function. Point the current node to the
   * parent node of its parent node.
   *
   * @param[in] node Vertex Id
   * @param[in] problem Data slice object
   * @param[in] v Vertex value
   * @param[in] nid Node ID
   */
  static __device__ __forceinline__ void ApplyFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    VertexId from = _ldg(d_data_slice->froms + node);
    VertexId to = _ldg(d_data_slice->tos + node);
    Value l = _ldg(d_data_slice->labels + from);
    atomicAdd(&d_data_slice->weight_reg[l], -d_data_slice->degrees[to]);
  }
};

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Structure contains device functions in label propagation.
 *
 * @tparam VertexId    Type of signed integer use as vertex identifier
 * @tparam SizeT       Type of integer / unsigned integer for array indexing
 * @tparam Value       Type of integer / float / double to attributes
 * @tparam ProblemData Problem data type contains data slice for MST problem
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct LpAssignWeightFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ bool CondFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    return true;
  }

  /**
   * @brief Filter Kernel apply function. Point the current node to the
   * parent node of its parent node.
   *
   * @param[in] node Vertex Id
   * @param[in] problem Data slice object
   * @param[in] v Vertex value
   * @param[in] nid Node ID
   */
  static __device__ __forceinline__ void ApplyFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    VertexId to = _ldg(d_data_slice->tos + node);
    VertexId l = _ldg(d_data_slice->labels + to);
    Value w_reg = _ldg(d_data_slice->weight_reg + l);
    d_data_slice->final_weights[node] =
        _ldg(d_data_slice->edge_weights + node) * w_reg * l;
  }
};

////////////////////////////////////////////////////////////////////////////////
/**
 * @brief Structure contains device functions in label propagation.
 *
 * @tparam VertexId    Type of signed integer use as vertex identifier
 * @tparam SizeT       Type of integer / unsigned integer for array indexing
 * @tparam Value       Type of integer / float / double to attributes
 * @tparam ProblemData Problem data type contains data slice for MST problem
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct LpSwapLabelFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ bool CondFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    return true;
  }

  /**
   * @brief Filter Kernel apply function. Point the current node to the
   * parent node of its parent node.
   *
   * @param[in] node Vertex Id
   * @param[in] problem Data slice object
   * @param[in] v Vertex value
   * @param[in] nid Node ID
   */
  static __device__ __forceinline__ void ApplyFilter(
      // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0)
      VertexId v, VertexId node, DataSlice *d_data_slice, SizeT nid,
      LabelT label, SizeT input_pos, SizeT output_pos) {
    cub::KeyValuePair<SizeT, Value> argmax = d_data_slice->argmax_kv[node];
    VertexId l = d_data_slice->labels[node];
    SizeT offset = _ldg(d_data_slice->offsets + node);
    VertexId new_l = _ldg(d_data_slice->tos + offset + argmax.key);
    if (l != new_l) {
      d_data_slice->stable_flag[0] = 0;
      d_data_slice->labels[node] = new_l;
    }
  }
};

}  // namespace lp
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
