// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * mst_functor.cuh
 *
 * @brief Device functions for MST problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/mst/mst_problem.cuh>

namespace gunrock {
namespace app {
namespace mst {

/**
 * @brief Structure contains device functions in MST graph traverse.
 * Used for finding flag array
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for BFS problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct FLAGFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. 
     * Used for generating flag array and successor array
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
        if (problem->d_reducedWeights[s_id] == problem->d_weights[e_id]){
                problem->d_successor[s_id] = d_id;
        }
	// TODO: 
	// problem->d_successor[s_id] = (problem->d_reducedWeights[s_id] == problem->d_weights[e_id]) ? d_id : problem->d_flag[e_id];
	return true;
    }

    /**
     * @brief Forward Edge Mapping apply function. 
     * Do nothing here
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
	return;
    }

    /**
     * @ set the flags[row_offset[vid]] = 1
     * Used for marking segmentations
     *
     * @ param[in] node Vertex Id
     * @ param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
	problem->d_flag[problem->d_row_offsets[node]] = 1;
    	problem->d_flag[0] = 0;
	return true;
    }

    /**
     * @brief Vertex mapping apply function. 
     *	Doing nothing here
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
	return;
    }
};

/**
 * @brief Structure contains device functions in MST graph traverse.
 *  Used for removing cycles in the graph
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for BFS problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct RCFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. 
     * Used for finding Vetex Id that have min weight
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id)
    {
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
"mst_functor.cuh" 102L, 3147C                                                                                                     1,1           Top
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
	return;
    }

    /**
     * @ set the Successor[node] = node if Successor[Successor[node]] = node 
     *
     * @ param[in] node Vertex Id
     * @ param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        problem->d_successor[node] = (problem->d_successor[problem->d_successor[node]] == node) ? ((problem->d_successor[node] > node)? node:problem->d_successor[node]) : problem->d_successor[node];
	return true;
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for BFS problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
	return;
    }
};




/**
 * @brief Structure contains device functions for pointer jumping operation.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for CC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PtrJumpFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Vertex mapping condition function. The vertex id is always valid.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        return true;
    }

    /**
     * @brief Vertex mapping apply function. Point the current node to the parent node
     * of its parent node.

     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        VertexId parent;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                parent, problem->d_represent + node);
        VertexId grand_parent;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                grand_parent, problem->d_represent + parent);
        if (parent != grand_parent) {
	
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        0, problem->d_vertex_flag); 
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        grand_parent, problem->d_represent + node);
        }
    }
};

/**
 * @brief Structure contains device functions for doing pointer jumping only for masked nodes.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for CC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PtrJumpMaskFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Vertex mapping condition function. The vertex id is always valid.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        return true; 
    }

    /**
     * @brief Vertex mapping apply function. Pointer jumping for the masked nodes. Point
     * the current node to the parent node of its parent node.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        VertexId mask;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                mask, problem->d_masks + node);
        if (mask == 0) {
            VertexId parent;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent, problem->d_represent + node);
            VertexId grand_parent;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    grand_parent, problem->d_represent + parent);
            if (parent != grand_parent) {
                problem->d_vertex_flag[0] = 0;
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        grand_parent, problem->d_represent + node);
            } else {
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        -1, problem->d_masks + node);
            }
        }
    }
};



/**
 * @brief Structure contains device functions in MST graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for MST problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct EdgeRmFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Edge removal 
     * 
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
	problem->d_edges[e_id] = (problem->d_represent[s_id] == problem->d_represent[d_id]) ? -1 : problem->d_edges[e_id]; 
	problem->d_weights[e_id] = (problem->d_represent[s_id] == problem->d_represent[d_id]) ? -1 : problem->d_weights[e_id];
	problem->d_keys[e_id] = (problem->d_represent[s_id] == problem->d_represent[d_id]) ? -1 : problem->d_keys[e_id];	
	// New flag for reduce length 
	problem->d_flag[e_id] = (problem->d_represent[s_id] == problem->d_represent[d_id]) ? -1 : problem->d_flag[e_id];
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
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
       return; 
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        problem->d_keys[node] = problem->d_Ckeys[problem->d_keys[node]];
	problem->d_edges[node] = problem->d_Ckeys[problem->d_edges[node]];
	return true;
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for BFS problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        return; 
    }
};




/**
 * @brief Structure contains device functions in MST graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for MST problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct RMFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
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
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
       return; 
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid.
     * Return true when key[v_id] == key[v_id-1].
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
   	return (node != -1) ? true : false;
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for BFS problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        return; 
    }
};

template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct VLENFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
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
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
       return; 
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        return true;
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for BFS problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        problem->d_row_offsets[node] = (problem->d_Cflag[node] == 0) ? -1 : 1;
	problem->d_row_offsets[0] = 1;
	return; 
    }
};





/**
 * @brief Structure contains device functions in MST graph traverse.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for MST problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct RowOFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Forward Edge Mapping condition function. Check if the destination node
     * has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge and include the destination node in the next frontier.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
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
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0)
    {
       return; 
    }

    /**
     * @brief Vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
	problem->d_row_offsets[0] = 0;
        if (problem->d_flag[node] == 1){
		problem->d_row_offsets[problem->d_keys[node]] = node;
	}
	//problem->d_row_offsets[problem->d_keys[node]] = (problem->d_flag[node] == 1) ? node : problem->d_row_offsets[problem->d_keys[node]];  
	return true;
    }

    /**
     * @brief Vertex mapping apply function. Doing nothing for BFS problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyVertex(VertexId node, DataSlice *problem, Value v = 0)
    {
        return; 
    }
};




} // mst
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
