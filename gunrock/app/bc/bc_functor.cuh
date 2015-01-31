// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_functor.cuh
 *
 * @brief Device functions for BC problem.
 */

#pragma once

#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/bc/bc_problem.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Structure contains device functions in forward traversal pass.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32
 * @tparam Value               Type of float or double to use for computing BC value.
 * @tparam ProblemData         Problem data type which contains data slice for BC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct ForwardFunctor
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
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        // Check if the destination node has been claimed as someone's child
        bool child_available = (atomicCAS(problem->preds + d_id, -2, s_id) == -2) ? true : false;
        
        if (!child_available)
        {
            //Two conditions will lead the code here.
            //1) multiple parents try to claim a same child,
            //and some parent other than you succeeded. In
            //this case the label of the child should be -1.
            //2) The child is from the same layer or maybe
            //the upper layer of the graph and it has been
            //labeled already.
            //We do an atomicCAS to make sure the child be
            //labeled.
            VertexId label;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    label, problem->labels + s_id);
            atomicCAS(problem->labels + d_id, -1, label+1);
            VertexId label_d;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    label_d, problem->labels + d_id);
            if (label_d == label + 1)
            {
                //Accumulate sigma value
                atomicAdd(problem->sigmas + d_id, problem->sigmas[s_id]);
            }
            return false;
        }
        else {
        return true;
        }
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
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    { 
            // Succeeded in claiming child, safe to set label to child
            VertexId label;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    label, problem->labels + s_id);
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                    label+1, problem->labels + d_id);
            atomicAdd(problem->sigmas + d_id, problem->sigmas[s_id]);
        
    }

    /**
     * @brief Forward vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return node != -1;
    }

    /**
     * @brief Forward vertex mapping apply function. Doing nothing for BC problem.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        // Doing nothing here
    }
};

/**
 * @brief Structure contains device functions in backward traversal pass.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value               Type of float or double to use for computing BC value.
 * @tparam ProblemData         Problem data type which contains data slice for BC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BackwardFunctor
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Backward Edge Mapping condition function. Check if the destination node
     * is the direct neighbor of the source node.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        
        VertexId s_label;
        VertexId d_label;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                s_label, problem->labels + s_id);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                d_label, problem->labels + d_id);
       return (d_label == s_label + 1);
    }

    /**
     * @brief Backward Edge Mapping apply function. Compute delta value using
     * the formula: delta(s_id) = sigma(s_id)/sigma(d_id)(1+delta(d_id))
     * then accumulate to BC value of the source node.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        //set d_labels[d_id] to be d_labels[s_id]+1
        Value from_sigma;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            from_sigma, problem->sigmas + s_id);

        Value to_sigma;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            to_sigma, problem->sigmas + d_id);

        Value to_delta;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            to_delta, problem->deltas + d_id);

        Value result = from_sigma / to_sigma * (1.0 + to_delta);

        //Accumulate delta value

        //Accumulate bc value
        //atomicAdd(problem->ebc_values + e_id, result);

        //printf("%d->%d : %f = %f / %f * (1.0 + %f)\n", s_id, d_id, result, from_sigma, to_sigma, to_delta);
        if (s_id != problem->src_node[0]) {
            atomicAdd(problem->deltas + s_id, result); 
            atomicAdd(problem->bc_values + s_id, result);
        }
    }

    /**
     * @brief Backward vertex mapping condition function. Check if the Vertex Id is valid (equal to 0).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return problem->labels + node == 0;
    }

    /**
     * @brief Backward vertex mapping apply function. doing nothing here.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        // Doing nothing here
    }
};

/**
 * @brief Structure contains device functions in backward traversal pass.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam Value               Type of float or double to use for computing BC value.
 * @tparam ProblemData         Problem data type which contains data slice for BC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct BackwardFunctor2
{
    typedef typename ProblemData::DataSlice DataSlice;

    /**
     * @brief Backward Edge Mapping condition function. Check if the destination node
     * is the direct neighbor of the source node.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the edge.
     */
    static __device__ __forceinline__ bool CondEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        
        VertexId s_label;
        VertexId d_label;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                s_label, problem->labels + s_id);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                d_label, problem->labels + d_id);
       return (d_label == s_label + 1);
    }

    /**
     * @brief Backward Edge Mapping apply function. Compute delta value using
     * the formula: delta(s_id) = sigma(s_id)/sigma(d_id)(1+delta(d_id))
     * then accumulate to BC value of the source node.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyEdge(VertexId s_id, VertexId d_id, DataSlice *problem, VertexId e_id = 0, VertexId e_id_in = 0)
    {
        //set d_labels[d_id] to be d_labels[s_id]+1
        Value from_sigma;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            from_sigma, problem->sigmas + s_id);

        Value to_sigma;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            to_sigma, problem->sigmas + d_id);

        Value to_delta;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
            to_delta, problem->deltas + d_id);

        //Value result = from_sigma / to_sigma * (1.0 + to_delta);

        //Accumulate delta value

        //Accumulate bc value
        //atomicAdd(problem->ebc_values + e_id, result);
        
        /*if (s_id != problem->d_src_node[0]) {
            atomicAdd(&problem->d_deltas[s_id], result); 
            atomicAdd(&problem->d_bc_values[s_id], result);
        }*/
    }

    /**
     * @brief Backward vertex mapping condition function. Check if the Vertex Id is valid (equal to 0).
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return problem->labels[node] == 0;
    }

    /**
     * @brief Backward vertex mapping apply function. doing nothing here.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        // Doing nothing here
    }
};

} // bc
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
