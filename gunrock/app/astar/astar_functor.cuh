// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * astar_functor.cuh
 *
 * @brief Device functions for A* problem.
 */

#pragma once
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/astar/astar_problem.cuh>
#include <stdio.h>

namespace gunrock {
namespace app {
namespace astar {

/**
 * @brief Structure contains device functions in A* path finding.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct ASTARFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ float GraphHeuristicFunc(
      VertexId cur_vid, DataSlice *d_data_slice) {
    VertexId dst_node = _ldg(&d_data_slice->dst_node[0]);
    Value sample_weight = _ldg(&d_data_slice->sample_weight[0]);
    VertexId levels = abs(_ldg(d_data_slice->bfs_levels + cur_vid) -
                          _ldg(d_data_slice->bfs_levels + dst_node));
    return levels * sample_weight;
  }

  /**
   * @brief Forward Edge Mapping condition function. Check if the destination
   * node has been claimed as someone else's child.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id output edge id
   * @param[in] e_id_in input edge id
   *
   * \return Whether to load the apply function for the edge and include the
   * destination node in the next frontier.
   */
  static __device__ __forceinline__ bool CondEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos)
  // VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
  // VertexId e_id = 0, VertexId e_id_in = 0)
  {
    Value pred_distance, edge_weight;

    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        pred_distance, d_data_slice->g_cost + s_id);
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        edge_weight, d_data_slice->weights + edge_id);
    Value new_distance = pred_distance + edge_weight;

    // Check if the destination node has been claimed as someone's child
    Value old_distance = atomicMin(d_data_slice->g_cost + d_id, new_distance);
    // if (to_track(s_id) || to_track(d_id))
    // printf("lable[%d] : %d -> %d @ %d + %d @ %d = %d \n", d_id, old_distance,
    // edge_weight, edge_id, pred_distance, s_id, new_distance);
    bool result = (new_distance < old_distance);

    return result;
  }

  /**
   * @brief Forward Edge Mapping apply function. Now we know the source node
   * has succeeded in claiming child, so it is safe to set distance to its child
   * node (destination node).
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id output edge id
   * @param[in] e_id_in input edge id
   *
   */
  static __device__ __forceinline__ void ApplyEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos)
  // VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
  // VertexId e_id = 0, VertexId e_id_in = 0)
  {
    if (Problem::MARK_PATHS)  //(Problem::MARK_PREDECESSORS)
    {
      if (d_data_slice->original_vertex.GetPointer(util::DEVICE) != NULL)
        s_id = d_data_slice->original_vertex[s_id];
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          s_id, d_data_slice->preds + d_id);
    }
  }

  /**
   * @brief Vertex mapping condition function. Check if the Vertex Id is valid
   * (not equal to -1).
   *
   * @param[in] node Vertex identifier.
   * @param[in] problem Data slice object.
   * @param[in] v auxiliary value.
   * @param[in] nid Vertex index.
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos)
  // VertexId node, DataSlice *d_data_slice, VertexId v = 0, SizeT nid = 0)
  {
    if (node == -1) return false;

    if (d_data_slice->labels[node] == label) {
      return false;
    }
    d_data_slice->labels[node] = label;
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Doing nothing for astar problem.
   *
   * @param[in] node Vertex identifier.
   * @param[in] problem Data slice object.
   * @param[in] v auxiliary value.
   * @param[in] nid Vertex index.
   *
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos)
  // VertexId node, DataSlice *d_data_slice, VertexId v = 0, SizeT nid = 0)
  {
    VertexId dst_node = _ldg(&d_data_slice->dst_node[0]);
    /*if (dst_node == node && d_data_slice->quit_flag[0] == 0)
        d_data_slice->quit_flag[0] = 1;*/
    // compute heuristic func
    Value h = GraphHeuristicFunc(node, d_data_slice);
    d_data_slice->f_cost[node] = d_data_slice->g_cost[node] + h;
  }
};

/**
 * @brief Structure contains device functions in A* path finding.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct ASTARDistanceFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  static __device__ __forceinline__ float DistanceHeuristicFunc(
      VertexId cur_vid, DataSlice *d_data_slice) {
    VertexId dst_node = _ldg(&d_data_slice->dst_node[0]);
    Value pointx2 = _ldg(d_data_slice->pointx + dst_node);
    Value pointy2 = _ldg(d_data_slice->pointy + dst_node);
    Value pointx1 = _ldg(d_data_slice->pointx + cur_vid);
    Value pointy1 = _ldg(d_data_slice->pointy + cur_vid);
    return sqrt((pointx1 - pointx2) * (pointx1 - pointx2) +
                (pointy1 - pointy2) * (pointy1 - pointy2));
  }

  /**
   * @brief Forward Edge Mapping condition function. Check if the destination
   * node has been claimed as someone else's child.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id output edge id
   * @param[in] e_id_in input edge id
   *
   * \return Whether to load the apply function for the edge and include the
   * destination node in the next frontier.
   */
  static __device__ __forceinline__ bool CondEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos)
  // VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
  // VertexId e_id = 0, VertexId e_id_in = 0)
  {
    Value pred_distance, edge_weight;

    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        pred_distance, d_data_slice->g_cost + s_id);
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        edge_weight, d_data_slice->weights + edge_id);
    Value new_distance = pred_distance + edge_weight;

    // Check if the destination node has been claimed as someone's child
    Value old_distance = atomicMin(d_data_slice->g_cost + d_id, new_distance);
    // if (to_track(s_id) || to_track(d_id))
    // printf("lable[%d] : %d -> %d @ %d + %d @ %d = %d \n", d_id, old_distance,
    // edge_weight, edge_id, pred_distance, s_id, new_distance);
    bool result = (new_distance < old_distance);
    // printf("s:%d d:%d e_id:%d, weight:%f, cond:%d\n", s_id, d_id, edge_id,
    // edge_weight, result); printf("distance_s:%f, distance_d:%f, weight:%f\n",
    // d_data_slice->g_cost[s_id], d_data_slice->g_cost[d_id], edge_weight);

    return result;
  }

  /**
   * @brief Forward Edge Mapping apply function. Now we know the source node
   * has succeeded in claiming child, so it is safe to set distance to its child
   * node (destination node).
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   * @param[in] e_id output edge id
   * @param[in] e_id_in input edge id
   *
   */
  static __device__ __forceinline__ void ApplyEdge(
      VertexId s_id, VertexId d_id, DataSlice *d_data_slice, SizeT edge_id,
      VertexId input_item, LabelT label, SizeT input_pos, SizeT &output_pos)
  // VertexId s_id, VertexId d_id, DataSlice *d_data_slice,
  // VertexId e_id = 0, VertexId e_id_in = 0)
  {
    if (Problem::MARK_PATHS)  //(Problem::MARK_PREDECESSORS)
    {
      if (d_data_slice->original_vertex.GetPointer(util::DEVICE) != NULL)
        s_id = d_data_slice->original_vertex[s_id];
      util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
          s_id, d_data_slice->preds + d_id);
      // printf("%d %d\n", s_id, d_id);
    }
  }

  /**
   * @brief Vertex mapping condition function. Check if the Vertex Id is valid
   * (not equal to -1).
   *
   * @param[in] node Vertex identifier.
   * @param[in] problem Data slice object.
   * @param[in] v auxiliary value.
   * @param[in] nid Vertex index.
   *
   * \return Whether to load the apply function for the node and include it in
   * the outgoing vertex frontier.
   */
  static __device__ __forceinline__ bool CondFilter(VertexId v, VertexId node,
                                                    DataSlice *d_data_slice,
                                                    SizeT nid, LabelT label,
                                                    SizeT input_pos,
                                                    SizeT output_pos)
  // VertexId node, DataSlice *d_data_slice, VertexId v = 0, SizeT nid = 0)
  {
    if (node == -1) return false;

    if (d_data_slice->labels[node] == label) {
      return false;
    }
    d_data_slice->labels[node] = label;
    return true;
  }

  /**
   * @brief Vertex mapping apply function. Doing nothing for astar problem.
   *
   * @param[in] node Vertex identifier.
   * @param[in] problem Data slice object.
   * @param[in] v auxiliary value.
   * @param[in] nid Vertex index.
   *
   */
  static __device__ __forceinline__ void ApplyFilter(VertexId v, VertexId node,
                                                     DataSlice *d_data_slice,
                                                     SizeT nid, LabelT label,
                                                     SizeT input_pos,
                                                     SizeT output_pos)
  // VertexId node, DataSlice *d_data_slice, VertexId v = 0, SizeT nid = 0)
  {
    // printf("node: %d\n", node);
    VertexId dst_node = _ldg(&d_data_slice->dst_node[0]);
    /*if (dst_node == node && d_data_slice->quit_flag[0] == 0)
        d_data_slice->quit_flag[0] = 1;*/
    // compute heuristic func
    Value h = DistanceHeuristicFunc(node, d_data_slice);
    d_data_slice->f_cost[node] = d_data_slice->g_cost[node] + h;
  }
};

template <typename VertexId, typename SizeT, typename Value, typename Problem,
          typename _LabelT = VertexId>
struct PQFunctor {
  typedef typename Problem::DataSlice DataSlice;
  typedef _LabelT LabelT;

  /**
   * @brief Forward Edge Mapping condition function. Check if the destination
   * node has been claimed as someone else's child.
   *
   * @param[in] s_id Vertex Id of the edge source node
   * @param[in] d_id Vertex Id of the edge destination node
   * @param[in] problem Data slice object
   *
   * \return Whether to load the apply function for the edge and include the
   * destination node in the next frontier.
   */
  static __device__ __forceinline__ Value
  ComputePriorityScore(VertexId node_id, DataSlice *d_data_slice) {
    Value distance;
    util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        distance, d_data_slice->f_cost + node_id);
    float delta = _ldg(&d_data_slice->delta[0]);
    return (delta == 0) ? distance : distance / delta;
  }
};

}  // namespace astar
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
