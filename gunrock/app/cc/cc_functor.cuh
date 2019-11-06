#pragma once

#include <stdio.h>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>
#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace app {
namespace cc {

/**
 * @brief Structure contains device functions for doing mask update.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 * @tparam _LabelT     Vertex label type.
 *
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct UpdateMaskFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;
  /**
   * @brief Vertex mapping condition function. The vertex id is always valid.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
    return true;
  }

  /**
   * @brief Vertex mapping apply function. If the component id equals to the
   * node id, set mask to 0, else set mask to 1.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
    VertexId parent;
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        parent, d_data_slice->component_ids + node);
    // if (TO_TRACK)
    //    if (to_track(node))
    //        printf("UpdateMask [%d]: %d->%d\n", node, d_data_slice ->
    //        masks[node], (parent == node) ? 0 : 1);
    // util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
    //    (parent == node) ? 0 : 1, d_data_slice->masks + node);
    d_data_slice->masks[node] = (parent == node) ? 0 : 1;
  }
};

/**
 * @brief Structure contains device functions for initialization of hook
 * operation.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 * @tparam _LabelT     Vertex label type.
 *
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct HookInitFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;
  /**
   * @brief Vertex mapping condition function. The vertex id is always valid.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Initialization of the hook operation.
   * Set the component id of the node which has the min node id to the max node
   * id.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    VertexId from_node = d_data_slice->froms[node];
    VertexId to_node = d_data_slice->tos[node];
    /*util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            from_node, problem->froms + node);
    util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            to_node, problem->tos + node);*/
    VertexId max_node = from_node > to_node ? from_node : to_node;
    VertexId min_node = from_node + to_node - max_node;
    // if (TO_TRACK)
    //    if (to_track(max_node) || to_track(min_node))
    //        printf("HookInit [%d]: ->%d\n", max_node, min_node);
    // util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
    //        min_node, problem->component_ids + max_node);
    d_data_slice->component_ids[max_node] = min_node;
  }

  /**
   * @brief impement HookInitFunctor using advance instead of filter
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[out] d_data_slice Data slice object.
   * @param[in] edge_id Edge index in the output frontier
   * @param[in] input_item Input Vertex Id
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[out] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the edge and include the
   * destination node in the next frontier.
   */
  static __device__ __forceinline__ bool CondEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos) {
    VertexId max_node = s_id > d_id ? s_id : d_id;
    VertexId min_node = s_id > d_id ? d_id : s_id;
    d_data_slice->component_ids[max_node] = min_node;
    return false;
  }

  /**
   * @brief impement HookInitFunctor using advance instead of filter
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[out] d_data_slice Data slice object.
   * @param[in] edge_id Edge index in the output frontier
   * @param[in] input_item Input Vertex Id
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[out] output_pos Index in the output frontier
   *
   */

  static __device__ __forceinline__ void ApplyEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos) {
    // nothing here
  }
};

/**
 * @brief Structure contains device functions for doing hook max node to min
 * node operation.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 *
 * @tparam _LabelT     Vertex label type.
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct HookMinFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  /**
   * @brief Vertex mapping condition function. The vertex id is always valid.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Compute the hook operation. Set the
   * component id of the node which has the min node id to the max node id.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
    bool mark;
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        mark, d_data_slice->marks + node);
    if (!mark) {
      VertexId from_node;
      VertexId to_node;
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          from_node, d_data_slice->froms + node);
      // from_node = _ldg(d_data_slice -> froms + node);
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          to_node, d_data_slice->tos + node);
      // to_node = _ldg(d_data_slice -> tos + node);
      VertexId parent_from;
      VertexId parent_to;
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          parent_from, d_data_slice->component_ids + from_node);
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          parent_to, d_data_slice->component_ids + to_node);
      VertexId min_node = parent_from <= parent_to ? parent_from : parent_to;
      VertexId max_node = parent_from + parent_to - min_node;
      if (max_node == min_node) {
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            true, d_data_slice->marks + node);
      } else {
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            max_node, d_data_slice->component_ids + min_node);
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            0, d_data_slice->edge_flag + 0);
      }
    }
  }
};

/**
 * @brief Structure contains device functions for doing hook min node to max
 * node operation.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 *
 * @tparam _LabelT     Vertex label type.
 *
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct HookMaxFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  /**
   * @brief Vertex mapping condition function. The vertex id is always valid.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    // VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Compute the hook operation. Set the
   * component id of the node which has the max node id to the min node id.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    bool mark;
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        mark, d_data_slice->marks + node);
    if (!mark) {
      VertexId from_node;
      VertexId to_node;
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          from_node, d_data_slice->froms + node);
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          to_node, d_data_slice->tos + node);
      VertexId parent_from;
      VertexId parent_to;
      parent_from = _ldg(d_data_slice->component_ids + from_node);
      parent_to = _ldg(d_data_slice->component_ids + to_node);
      if (parent_from == parent_to) {
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            true, d_data_slice->marks + node);
      } else {
        VertexId max_node = parent_from > parent_to ? parent_from : parent_to;
        VertexId min_node = parent_from + parent_to - max_node;
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            min_node, d_data_slice->component_ids + max_node);
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            0, d_data_slice->edge_flag + 0);
      }
    }
  }

  /**
   * @brief implement HookMaxFunctor using advance instead of filter
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[out] d_data_slice Data slice object.
   * @param[in] edge_id Edge index in the output frontier
   * @param[in] input_item Input Vertex Id
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[out] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the edge and include the
   * destination node in the next frontier.
   */
  static __device__ __forceinline__ bool CondEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos) {
    bool mark;
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        mark, d_data_slice->marks + edge_id);
    if (mark) return false;

    VertexId parent_from;
    VertexId parent_to;
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        parent_from, d_data_slice->component_ids + s_id);
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        parent_to, d_data_slice->component_ids + d_id);
    VertexId max_node = parent_from > parent_to ? parent_from : parent_to;
    VertexId min_node = parent_from > parent_to ? parent_to : parent_from;

    if (max_node == min_node) {
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          true, d_data_slice->marks + /*node*/ edge_id);
    } else {
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          min_node, d_data_slice->component_ids + max_node);
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          0, d_data_slice->edge_flag + 0);
    }
    return false;
  }

  /**
   * @brief implement HookMaxFunctor using advance instead of filter
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[out] d_data_slice Data slice object.
   * @param[in] edge_id Edge index in the output frontier
   * @param[in] input_item Input Vertex Id
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[out] output_pos Index in the output frontier
   *
   */
  static __device__ __forceinline__ void ApplyEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos) {
    // nothing here
  }
};

/**
 * @brief Structure contains device functions for pointer jumping operation.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 *
 * @tparam _LabelT     Vertex label type.
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct PtrJumpFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;
  /**
   * @brief Vertex mapping condition function. The vertex id is always valid.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Point the current node to the parent
   * node of its parent node.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    VertexId parent;
    parent = _ldg(d_data_slice->component_ids + node);
    VertexId grand_parent;
    grand_parent = _ldg(d_data_slice->component_ids + parent);
    if (parent != grand_parent) {
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          0, d_data_slice->vertex_flag + 0);
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          grand_parent, d_data_slice->component_ids + node);
    }
  }
};

/**
 * @brief Structure contains device functions for doing pointer jumping only for
 * masked nodes.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 *
 * @tparam _LabelT     Vertex label type.
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct PtrJumpMaskFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;
  /**
   * @brief Vertex mapping condition function. The vertex id is always valid.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Pointer jumping for the masked nodes.
   * Point the current node to the parent node of its parent node.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    if (d_data_slice->masks[node] == 0) {
      VertexId parent;
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          parent, d_data_slice->component_ids + node);
      VertexId grand_parent;
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          grand_parent, d_data_slice->component_ids + parent);
      if (parent != grand_parent) {
        d_data_slice->vertex_flag[0] = 0;
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            grand_parent, d_data_slice->component_ids + node);
      } else {
        d_data_slice->masks[node] = -1;
      }
    }
  }
};

/**
 * @brief Structure contains device functions for doing pointer jumping for
 * unmasked nodes.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam Problem     Problem data type which contains data slice for problem.
 *
 * @tparam _LabelT     Vertex label type.
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct PtrJumpUnmaskFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  /**
   * @brief Vertex mapping condition function. The vertex id is always valid.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos) {
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Pointer jumping for the unmasked
   * nodes. Point the current node to the parent node of its parent node.
   *
   * @param[in] v auxiliary value.
   * @param[in] node Vertex identifier.
   * @param[out] d_data_slice Data slice object.
   * @param[in] nid Vertex index.
   * @param[in] label Vertex label value.
   * @param[in] input_pos Index in the input frontier
   * @param[in] output_pos Index in the output frontier
   *
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos) {
    if (d_data_slice->masks[node] == 1) {
      VertexId parent;
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          parent, d_data_slice->component_ids + node);
      VertexId grand_parent;
      util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
          grand_parent, d_data_slice->component_ids + parent);
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          grand_parent, d_data_slice->component_ids + node);
    }
  }
};

}  // namespace cc
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
