// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * frontier.cuh
 *
 * @brief Defination of frontier
 */

#pragma once

#include <gunrock/util/array_utils.cuh>

namespace gunrock {
namespace app {

/**
 * @brief Enumeration of global frontier queue configurations
 */
enum FrontierType
{
    VERTEX_FRONTIER,       // O(|V|) ping-pong global vertex frontier
    EDGE_FRONTIER,         // O(|E|) ping-pong global edge frontier
};

/**
 * @brief Structure for frontier
 */
template <
    typename VertexT,
    typename SizeT = VertexT,
    util::ArrayFlag FLAG = util::ARRAY_NONE,
    unsigned int cudaHostRegisterFlag = cudaHostRegisterFlag>
struct Frontier
{
    std::string  frontier_name;
    util::CtaWorkProgressLifetime<SizeT> work_progress; // Queue size counters
    SizeT        queue_length; // the length of the current queue
    unsigned int queue_index ; // the index of the current queue
    bool         queue_reset ; // whether to reset the next queue

    unsigned int num_queues  ; // how many queues to support
    unsigned int num_vertex_queues; // num of vertex queues
    unsigned int num_edge_queues; // num of edge queues
    util::Array1D<SizeT, FrontierType, FLAG, cudaHostRegisterFlag>  queue_types; // types of each queue
    util::Array1D<SizeT, unsigned int, FLAG, cudaHostRegisterFlag >  queue_map; // mapping queue index to vertex_queue / edge_queue indices

    util::Array1D<SizeT, SizeT, FLAG | util::PINNED,
        cudaHostRegisterFlag | cudaHostAllocMapped | cudaHostAllocPortable>  output_length;

    util::Array1D<SizeT, SizeT        , FLAG, cudaHostRegisterFlag>  num_segments; // how many segments of each queue
    util::Array1D<SizeT, SizeT        , FLAG, cudaHostRegisterFlag> *segment_offsets; // offsets of segments for each queue

    util::Array1D<SizeT, SizeT        , FLAG, cudaHostRegisterFlag>  queue_offsets; //

    util::Array1D<SizeT, SizeT        , FLAG, cudaHostRegisterFlag>  scanned_edges           ; // length / offsets for offsets of the frontier queues
    util::Array1D<SizeT, unsigned char, FLAG, cudaHostRegisterFlag>  cub_scan_space;

    //Frontier queues. Used to track working frontier.
    util::Array1D<SizeT, VertexT, FLAG, cudaHostRegisterFlag> *vertex_queues; // vertex queues
    util::Array1D<SizeT, SizeT  , FLAG, cudaHostRegisterFlag> *edge_queues; // edge queues

    Frontier()
    {
        SetName("");
        segment_offsets = NULL;
        vertex_queues   = NULL;
        edge_queues     = NULL;
    }

    cudaError_t SetName(std::string name)
    {
        cudaError_t retval = cudaSuccess;

        frontier_name = name;
        if (name != "") name = name + "::";
        work_progress .SetName(name + "work_progress");
        queue_types   .SetName(name + "queue_types");
        queue_map     .SetName(name + "queue_map");
        output_length .SetName(name + "output_length");
        num_segments  .SetName(name + "num_segments");
        queue_offsets .SetName(name + "queue_offsets");
        scanned_edges .SetName(name + "scanned_edges");
        cub_scan_space.SetName(name + "cub_scan_space");
        return retval;
    }

    cudaError_t Init(
        unsigned int num_queues = 2,
        FrontierType *types = NULL,
        std::string frontier_name = "",
        util::Location target = util::DEVICE)
    {
        cudaError_t retval = cudaSuccess;
        retval = SetName(frontier_name);
        if (retval) return retval;

        this -> num_queues = num_queues;
        retval = queue_types.Allocate(num_queues, util::HOST);
        if (retval) return retval;
        retval = queue_map  .Allocate(num_queues, util::HOST);
        if (retval) return retval;
        num_vertex_queues = 0;
        num_edge_queues = 0;
        for (unsigned int q = 0; q < num_queues; q++)
        {
            FrontierType queue_type = VERTEX_FRONTIER;
            if (types != NULL || types[q] == EDGE_FRONTIER)
                queue_type = EDGE_FRONTIER;

            queue_types[q] = queue_type;
            if (queue_type == VERTEX_FRONTIER)
            {
                queue_map[q] = num_vertex_queues;
                num_vertex_queues ++;
            } else if (queue_type == EDGE_FRONTIER)
            {
                queue_map[q] = num_edge_queue;
                num_edge_queues ++;
            }
        }

        if (frontier_name != "")
            frontier_name = frontier_name + "::";
        retval = num_segments.Allocate(num_queues, util::HOST | target);
        if (retval) return retval;
        segment_offsets = new util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag>[num_queues];
        vertex_queues = new util::Array1D<SizeT, VertexT, FLAG, cudaHostRegisterFlag>[num_vertex_queues];
        edge_queues = new util::Array1D<SizeT, SizeT, FLAG, cudaHostRegisterFlag>[num_edge_queues];
        for (unsigned int q = 0; q < num_queues; q++)
        {
            segment_offsets[q].SetName(frontier_name
                + "segment_offsets["
                + std::to_string(q) + "]");
            if (queue_types[q] == VERTEX_FRONTIER)
            {
                auto &queue = vertex_queues[queue_map[q]];
                queue.SetName(frontier_name
                    + "queues[" + std::to_string(q) + "]");
            } else if (queue_types[q] == EDGE_FRONTIER)
            {
                auto &queue = vertex_queues[queue_map[q]];
                queue.SetName(frontier_name
                    + "queues[" + std::to_string(q) + "]");
            }
        }

        retval = output_length.Allocate(1, target | util::HOST);
        if (retval) return retval;
        retval = queue_offsets.Allcoate(num_queues, target | util::HOST);
        if (retval) return retval;
        return retval;
    }

    cudaError_t Release()
    {
        cudaError_t retval = cudaSuccess;

        for (unsigned int q = 0; q < num_queues; q++)
        {
            retval = segment_offsets[q].Release();
            if (retval) return retval;

            if (queue_types[q] == VERTEX_FRONTIER)
            {
                auto queue = vertex_queues[queue_map[q]];
                retval = queue.Release();
                if (retval) return retval;
            } else if (queue_types[q] == EDGE_FRONTIER)
            {
                auto queue = edge_queues[queue_map[q]];
                retval = queue.Release();
                if (retval) return retval;
            }
        }

        if (retval = queue_types  .Release()) return retval;
        if (retval = queue_map    .Release()) return retval;
        if (retval = output_length.Release()) return retval;
        if (retval = num_segments .Release()) return retval;
        if (retval = queue_offsets.Release()) return retval;
        if (retval = scanned_edges.Release()) return retval;
        if (retval = cub_scan_space.Release()) return retval;
        delete[] segment_offsets; segment_offsets = NULL;
        delete[] vertex_queues  ; vertex_queues   = NULL;
        delete[] edge_queues    ; edge_queues     = NULL;
        return retval;
    }

}; // struct Frontier

} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
