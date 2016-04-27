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
#include <gunrock/util/device_intrinsics.cuh>

namespace gunrock {
namespace app {
namespace bc {

/**
 * @brief Structure contains device functions in forward traversal pass.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId >
struct ForwardFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Forward Edge Mapping condition function. Check if the
     * destination node has been claimed as someone else's child.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether or not to load the apply function.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
        //VertexId s_id, VertexId d_id, DataSlice *problem,
        //VertexId e_id = 0, VertexId e_id_in = 0) {
    {
        // Check if the destination node has been claimed as someone's child
        //VertexId pred = s_id;
        //if (d_data_slice -> original_vertex.GetPointer(util::DEVICE) != NULL)
        //    pred = d_data_slice -> original_vertex[s_id];
        //bool child_available =
        //    (atomicCAS(d_data_slice -> labels + d_id, -1, label) == -1) ? true : false;
        VertexId old_label = atomicCAS(d_data_slice -> labels + d_id, -1, label);
        //VertexId old_label = d_data_slice -> labels[d_id];
        //if (old_label == -1) d_data_slice -> labels[d_id] = label;
        //if (old_label == -1) return true;
        if (old_label != label && old_label != -1) return false;

        //if (!child_available) 
        //if (old_label == -1)
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
            //VertexId label;
            //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            //    label, d_data_slice ->labels + s_id);
            
            //atomicCAS(d_data_slice ->labels + d_id, -1, label /*+ 1*/);
            //VertexId label_d;
            //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            //    label_d, d_data_slice->labels + d_id);
            //label_d = _ldg(d_data_slice -> labels + d_id);
            //if (label_d == label /*+ 1*/) {
                //Accumulate sigma value
                atomicAdd(d_data_slice->sigmas + d_id, d_data_slice->sigmas[s_id]);
            //}
            if (old_label == -1) 
            {
                return true;
            }
            else return false;
        } //else {
          //  return true;
        //}
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
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        //VertexId s_id, VertexId d_id, DataSlice *problem,
        //VertexId e_id = 0, VertexId e_id_in = 0) {
        // Succeeded in claiming child, safe to set label to child
        //VertexId label_;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    label_, d_data_slice ->labels + s_id);
        //util::io::ModifiedStore<Problem::QUEUE_WRITE_MODIFIER>::St(
        //    label_ + 1, d_data_slice ->labels + d_id);
        //atomicAdd(d_data_slice->sigmas + d_id, d_data_slice->sigmas[s_id]);
    }

    /**
     * @brief Forward vertex mapping condition function. Check if the Vertex Id is valid (not equal to -1).
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     * \return Whether or not to load the apply function.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        //VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
        return node != -1;
    }

    /**
     * @brief Forward vertex mapping apply function. Doing nothing for BC.
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        //VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
        // Doing nothing here
    }
};

/**
 * @brief Structure contains device functions in backward traversal pass.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId >
struct BackwardFunctor {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Backward Edge Mapping condition function. Check if the destination node
     * is the direct neighbor of the source node.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        //VertexId s_id, VertexId d_id, DataSlice *problem,
        //VertexId e_id = 0, VertexId e_id_in = 0) {
        VertexId s_label;
        VertexId d_label;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    s_label, d_data_slice->labels + s_id);
        s_label = _ldg(d_data_slice -> labels + s_id);
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    d_label, d_data_slice->labels + d_id);
        d_label = _ldg(d_data_slice -> labels + d_id);
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
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        if (s_id == d_data_slice->src_node[0]) return;
        // VertexId s_id, VertexId d_id, DataSlice *problem,
        // VertexId e_id = 0, VertexId e_id_in = 0) {
        //set d_labels[d_id] to be d_labels[s_id]+1
        Value from_sigma;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    from_sigma, d_data_slice->sigmas + s_id);
        from_sigma = _ldg(d_data_slice -> sigmas + s_id);

        Value to_sigma;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    to_sigma, d_data_slice->sigmas + d_id);
        to_sigma = _ldg(d_data_slice -> sigmas + d_id);

        Value to_delta;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    to_delta, d_data_slice->deltas + d_id);
        to_delta = _ldg(d_data_slice -> deltas + d_id);

        Value result = from_sigma / to_sigma * (1.0 + to_delta);

        //Accumulate delta value

        //Accumulate bc value
        //atomicAdd(problem->ebc_values + e_id, result);

        //if (s_id != d_data_slice->src_node[0]) 
        {
            Value old_delta = atomicAdd(d_data_slice->deltas + s_id, result);
            Value old_bc_value = atomicAdd(d_data_slice->bc_values + s_id, result);
            /*if (d_data_slice -> original_vertex + 0 != NULL)
            {
                s_id = d_data_slice -> original_vertex[s_id];
                d_id = d_data_slice -> original_vertex[d_id];
            }
            printf("%2d -> %2d : result = %.4f / %.4f * (1.0 + %.4f)\n ",
                s_id, d_id, from_sigma, to_sigma, to_delta);
            printf("%2d -> %2d : delta[%2d] = %.4f + %.4f = %.4f\n",
                s_id, d_id, s_id, old_delta, result, old_delta + result);*/
        } //else printf("%2d -> %2d : skipped\n");
    }

    /**
     * @brief Backward vertex mapping condition function. Check if the Vertex Id is valid (equal to 0).
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     *
     * \return Whether to load the apply function for the node and include it in the outgoing vertex frontier.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        //VertexId node, DataSlice *problem, Value v = 0) {
        return d_data_slice->labels + node == 0;
    }

    /**
     * @brief Backward vertex mapping apply function. doing nothing here.
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        //VertexId node, DataSlice *problem, Value v = 0) {
        // Doing nothing here
    }
};

/**
 * @brief Structure contains device functions in backward traversal pass.
 *
 * @tparam VertexId    Type of signed integer to use as vertex identifier.
 * @tparam SizeT       Type of unsigned integer to use for array indexing.
 * @tparam Value       Type of float or double to use for computed values.
 * @tparam ProblemData Problem data type which contains data slice for problem.
 *
 */
template <
    typename VertexId, typename SizeT, typename Value, typename Problem, typename _LabelT = VertexId >
struct BackwardFunctor2 {
    typedef typename Problem::DataSlice DataSlice;
    typedef _LabelT LabelT;

    /**
     * @brief Backward Edge Mapping condition function. Check if the destination node
     * is the direct neighbor of the source node.
     *
     * @param[in] s_id Vertex Id of the edge source node
     * @param[in] d_id Vertex Id of the edge destination node
     * @param[in] problem Data slice object
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     * \return Whether to load the apply function for the edge.
     */
    static __device__ __forceinline__ bool CondEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        //VertexId s_id, VertexId d_id, DataSlice *problem,
        //VertexId e_id = 0, VertexId e_id_in = 0) {

        VertexId s_label;
        VertexId d_label;
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            s_label, d_data_slice->labels + s_id);
        util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
            d_label, d_data_slice->labels + d_id);
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
     * @param[in] e_id output edge id
     * @param[in] e_id_in input edge id
     *
     */
    static __device__ __forceinline__ void ApplyEdge(
        VertexId s_id,
        VertexId d_id,
        DataSlice *d_data_slice,
        SizeT    edge_id   ,   
        VertexId input_item,
        LabelT   label     ,   
        SizeT    input_pos ,
        SizeT   &output_pos)
    {
        //VertexId s_id, VertexId d_id, DataSlice *problem,
        //VertexId e_id = 0, VertexId e_id_in = 0) {
        //set d_labels[d_id] to be d_labels[s_id]+1
        //Value from_sigma;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    from_sigma, d_data_slice->sigmas + s_id);

        //Value to_sigma;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    to_sigma, d_data_slice->sigmas + d_id);

        //Value to_delta;
        //util::io::ModifiedLoad<Problem::COLUMN_READ_MODIFIER>::Ld(
        //    to_delta, d_data_slice->deltas + d_id);

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
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     * \return Whether or not to load the apply function.
     */
    static __device__ __forceinline__ bool CondFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        //VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
        return d_data_slice->labels[node] == 0;
    }

    /**
     * @brief Backward vertex mapping apply function. doing nothing here.
     *
     * @param[in] node Vertex identifier.
     * @param[in] problem Data slice object.
     * @param[in] v auxiliary value.
     * @param[in] nid Vertex index.
     *
     */
    static __device__ __forceinline__ void ApplyFilter(
        VertexId   v,
        VertexId   node,
        DataSlice *d_data_slice,
        SizeT      nid  ,
        LabelT     label,
        SizeT      input_pos,
        SizeT      output_pos)
    {
        //VertexId node, DataSlice *problem, Value v = 0, SizeT nid = 0) {
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
