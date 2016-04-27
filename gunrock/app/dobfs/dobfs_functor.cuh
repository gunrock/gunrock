// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * dobfs_functor.cuh
 *
 * @brief Device functions for DOBFS problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/dobfs/dobfs_problem.cuh>

namespace gunrock {
namespace app {
namespace dobfs {

// TODO:
// 
// Prepare for reverse BFS (first two functor set)
// 1) prepare unvisited queue
//   VertexMap for all nodes, select whose label is -1
// 2) prepare frontier_map_in
//   Use MemsetKernel to set all frontier_map_in as 0
//   Vertexmap for all the nodes in current frontier,
//   set their frontier_map_in value as 1
// 3) clear all frontier_map_in value as 0
//

/**
 * @brief Structure contains device functions for Reverse BFS Preparation
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for BFS problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
struct PrepareInputFrontierMapFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v auxiliary value
     * @param[in] nid node id
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(//VertexId node, DataSlice *problem, Value v =0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
       return true; 
    }

    /**
     * @brief Vertex mapping apply function. Set frontier_map_in
     */
    static __device__ __forceinline__ void ApplyFilter(//VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            true, d_data_slice->d_frontier_map_in + node);
    }
};

/**
 * @brief Structure contains device functions for Reverse BFS Preparation
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for BFS problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
struct PrepareUnvisitedQueueFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (label equals to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v auxiliary value
     * @param[in] nid node id
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(//VertexId node, DataSlice *problem, Value v =0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        VertexId new_label;
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            new_label, d_data_slice->labels + node);
        return (new_label == -1 || new_label == util::MaxValue<Value>()-1);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing.
     */
    static __device__ __forceinline__ void ApplyFilter(//VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        // Doing nothing here
    }
};

// During the reverse BFS (third functor set)
// 1) BackwardEdgeMap
// 2) Clear frontier_map_in
// 3) VertexMap
//
/**
 * @brief Structure contains device functions for Reverse BFS Preparation
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for BFS problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
struct ReverseBFSFunctor
{
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;
    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(//VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        // Check if the destination node has been claimed as someone's child
        //return (atomicCAS(&problem->d_preds[d_id], -2, s_id) == -2) ? true : false;
        if (Problem::MARK_PREDECESSORS)
            util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
                    s_id, d_data_slice->preds + d_id);
        return true; 
    }

    /**
     * @brief Forward Edge Mapping apply function. Now we know the source node
     * has succeeded in claiming child, so it is safe to set label to its child
     * node (destination node).
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     */
    static __device__ __forceinline__ void ApplyEdge(//VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,
        VertexId input_item,
        LabelT   label     ,
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        //set d_labels[d_id] to be d_labels[s_id]+1
        VertexId label = s_id;
        if (Problem::MARK_PREDECESSORS)
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            label, d_data_slice->labels + s_id);
        util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
            label+1, d_data_slice->labels + d_id);
        
        //printf("src:%d, dst:%d, label:%d\n", s_id, d_id, problem->d_labels[d_id]);
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equals to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v auxiliary value
     * @param[in] nid node id
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(//VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        return (node != -1);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing.
     */
    static __device__ __forceinline__ void ApplyFilter(//VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        // Doing nothing here
    }
};
//
// Switch back to normal BFS (final functor set)
// 1) prepare current frontier
// VertexMap for all nodes, select whose frontier_map_out is 1
//
/**
 * @brief Structure contains device functions for Switching back to normal BFS
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for BFS problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId>
struct SwitchToNormalFunctor
{
    typedef typename Problem::DataSlice DataSlice; 
    typedef _LabelT LabelT;

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (frontier_map_out is set).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     * @param[in] v auxiliary value
     * @param[in] nid node id
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(//VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        bool flag;
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            flag, d_data_slice->d_frontier_map_out + node);
        return (flag);
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing.
     */
    static __device__ __forceinline__ void ApplyFilter(//VertexId node, DataSlice *problem, Value v = 0, SizeT nid=0)
        VertexId   v,  
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        // Doing nothing here
    }
};

} // dobfs
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
