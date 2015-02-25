#pragma once

#include <stdio.h>
#include <gunrock/app/problem_base.cuh>
#include <gunrock/app/cc/cc_problem.cuh>

namespace gunrock {
namespace app {
namespace cc {

    template <typename VertexId>
    static __device__ bool to_track(VertexId node)
    {
        const int num_to_track = 4;
        const VertexId node_to_track[] = {39852, 60160, 60161, 60162};
        for (int i=0; i<num_to_track; i++)
            if (node == node_to_track[i]) return true;
        return false;
    }
 
/**
 * @brief Structure contains device functions for doing mask update.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for CC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct UpdateMaskFunctor
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
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return true;
    }

    /**
     * @brief Vertex mapping apply function. If the component id equals to the node id, set mask
     * to 0, else set mask to 1.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        VertexId parent;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                parent, problem->component_ids + node);
        //if (to_track(node)) 
        //    printf("UpdateMask [%d]: %d->%d\n", node, problem->masks[node], (parent == node)?0:1);
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                (parent == node)?0:1, problem->masks + node);
   }
};

/**
 * @brief Structure contains device functions for initialization of hook operation.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for CC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookInitFunctor
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
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return true; 
    }

    /**
     * @brief Vertex mapping apply function. Initialization of the hook operation. Set the component id
     * of the node which has the min node id to the max node id.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        VertexId from_node;
        VertexId to_node;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                from_node, problem->froms + node);
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                to_node, problem->tos + node);
        VertexId max_node = from_node > to_node ? from_node:to_node;
        VertexId min_node = from_node + to_node - max_node;
        //if (to_track(max_node) || to_track(min_node))
        //    printf("HookInit [%d]: ->%d\n", max_node, min_node);
        util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                min_node, problem->component_ids + max_node);
    }
};

/**
 * @brief Structure contains device functions for doing hook max node to min node operation.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for CC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookMinFunctor
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
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return true;
    }

    /**
     * @brief Vertex mapping apply function. Compute the hook operation. Set the component id
     * of the node which has the min node id to the max node id.

     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        bool mark;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                mark, problem->marks + node);
        if (!mark) {
            VertexId from_node;
            VertexId to_node;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    from_node, problem->froms + node);
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    to_node, problem->tos + node);
            VertexId parent_from;
            VertexId parent_to;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent_from, problem->component_ids + from_node);
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent_to, problem->component_ids + to_node);
            VertexId max_node = parent_from > parent_to ? parent_from: parent_to;
            VertexId min_node = parent_from + parent_to - max_node;
            if (max_node == min_node) {
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        true, problem->marks + node);
            } else {
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        max_node, problem->component_ids + min_node);
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        0, problem->edge_flag +0);
            }
        }
    }
};

/**
 * @brief Structure contains device functions for doing hook min node to max node operation.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for CC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct HookMaxFunctor
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
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
       return true; 
    }

    /**
     * @brief Vertex mapping apply function. Compute the hook operation. Set the component id
     * of the node which has the max node id to the min node id.

     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        bool mark;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                mark, problem->marks + node);
        if (!mark) 
        {
            VertexId from_node;
            VertexId to_node;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    from_node, problem->froms + node);
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    to_node, problem->tos + node);
            VertexId parent_from;
            VertexId parent_to;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent_from, problem->component_ids + from_node);
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent_to, problem->component_ids + to_node);
            VertexId max_node = parent_from > parent_to ? parent_from: parent_to;
            VertexId min_node = parent_from + parent_to - max_node;
            if (max_node == min_node) {
                //if (to_track(max_node) || to_track(from_node) || to_track(to_node) || to_track(min_node)) 
                //    printf("HookMax n=%d, f_n=%d, t_n=%d, f_p=%d, t_p=%d: [%d] %d==\n", node, from_node, to_node, parent_from, parent_to, max_node, min_node); 
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        true, problem->marks + node);
            } else //if (problem->component_ids[max_node] > min_node) 
            {
                //if (to_track(max_node) || to_track(from_node) || to_track(to_node) || to_track(min_node)) 
                //    printf("HookMax n=%d, f_n=%d, t_n=%d, f_p=%d, t_p=%d: [%d] %d->%d\n", node, from_node, to_node, parent_from, parent_to, max_node, problem->component_ids[max_node], min_node); 
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        min_node, problem->component_ids + max_node);
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        0, problem->edge_flag + 0);
            }
        }
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
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
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
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        VertexId parent;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                parent, problem->component_ids + node);
        VertexId grand_parent;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                grand_parent, problem->component_ids + parent);
        if (parent != grand_parent) {
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        0, problem->vertex_flag + 0);
            //if (to_track(node)) 
            //    printf("PtrJump [%d]: %d->%d\n", node, problem->component_ids[node], grand_parent);
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        grand_parent, problem->component_ids + node);
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
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
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
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        VertexId mask;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                mask, problem->masks + node);
        if (mask == 0) {
            VertexId parent;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent, problem->component_ids + node);
            VertexId grand_parent;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    grand_parent, problem->component_ids + parent);
            if (parent != grand_parent) {
                problem->vertex_flag[0] = 0;
                //if (to_track(node)) 
                //    printf("PtrJumpMask [%d]: %d->%d\n", node, problem->component_ids[node], grand_parent);
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        grand_parent, problem->component_ids + node);
            } else {
                //if (to_track(node)) 
                //    printf("PtrJumpMask mask[%d]: %d->%d\n", node, problem->masks[node], -1);
                util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                        -1, problem->masks + node);
            }
        } //else if (to_track(node)) 
        //printf("PtrJumpMask mask[%d] = %d\n", node, mask);
    }
};

/**
 * @brief Structure contains device functions for doing pointer jumping for unmasked nodes.
 *
 * @tparam VertexId            Type of signed integer to use as vertex id (e.g., uint32)
 * @tparam SizeT               Type of unsigned integer to use for array indexing. (e.g., uint32)
 * @tparam ProblemData         Problem data type which contains data slice for CC problem
 *
 */
template<typename VertexId, typename SizeT, typename Value, typename ProblemData>
struct PtrJumpUnmaskFunctor
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
    static __device__ __forceinline__ bool CondFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        return true;
    }

    /**
     * @brief Vertex mapping apply function. Pointer jumping for the unmasked nodes. Point
     * the current node to the parent node of its parent node.
     *
     * @param[in] node Vertex Id
     * @param[in] problem Data slice object
     *
     */
    static __device__ __forceinline__ void ApplyFilter(VertexId node, DataSlice *problem, Value v = 0)
    {
        VertexId mask;
        util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                mask, problem->masks + node);
        if (mask == 1) {
            VertexId parent;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    parent, problem->component_ids + node);
            VertexId grand_parent;
            util::io::ModifiedLoad<ProblemData::COLUMN_READ_MODIFIER>::Ld(
                    grand_parent, problem->component_ids + parent);
            //if (to_track(node)) 
            //    printf("PtrJumpUnMask [%d]: %d->%d\t", node, parent, grand_parent);
            util::io::ModifiedStore<ProblemData::QUEUE_WRITE_MODIFIER>::St(
                    grand_parent, problem->component_ids + node);
        }
    }
};

} // cc
} // app
} // gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
