// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * bc_test.cu
 *
 * @brief Test related functions for BC
 */

#pragma once

#include <iostream>
#include <queue>
#include <vector>
#include <utility>

namespace gunrock {
namespace app {
namespace bc {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

// /**
//  * @brief Displays the SSSP result (i.e., distance from source)
//  * @tparam T Type of values to display
//  * @tparam SizeT Type of size counters
//  * @param[in] preds Search depth from the source for each node.
//  * @param[in] num_nodes Number of nodes in the graph.
//  */
// template<typename T, typename SizeT>
// void DisplaySolution(T *array, SizeT length)
// {
//     if (length > 40)
//         length = 40;

//     util::PrintMsg("[", true, false);
//     for (SizeT i = 0; i < length; ++i)
//     {
//         util::PrintMsg(std::to_string(i) + ":"
//             + std::to_string(array[i]) + " ", true, false);
//     }
//     util::PrintMsg("]");
// }

/******************************************************************************
 * Template Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference SSSP ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the distances
 * @param[in]   graph         Input graph
 * @param[out]  distances     Computed distances from the source to each vertex
 * @param[out]  preds         Computed predecessors for each vertex
 * @param[in]   src           The source vertex
 * @param[in]   quiet         Whether to print out anything to stdout
 * @param[in]   mark_preds    Whether to compute predecessor info
 * \return      double        Time taken for the SSSP
 */
 
template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT,
    typename VertexT = typename GraphT::VertexT>
double CPU_Reference(
    const GraphT &graph,
    ValueT *bc_values,
    ValueT *sigmas,
    VertexT *source_path,
    VertexT src,
    bool quiet)
{
        
    for(VertexT i = 0; i < graph.nodes; ++i) {
        bc_values[i]   = 0;
        sigmas[i]      = i == src ? 1 : 0;
        source_path[i] = i == src ? 0 : -1;
    }
    
    VertexT search_depth = 0;
    
    std::deque<VertexT> frontier;
    frontier.push_back(src);
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    while(!frontier.empty()) {
        VertexT dequeued_node = frontier.front();
        frontier.pop_front();
        VertexT neighbor_dist = source_path[dequeued_node] + 1;
        
        int edges_begin = graph.row_offsets[dequeued_node];
        int edges_end   = graph.row_offsets[dequeued_node + 1];
        
        for(int edge = edges_begin; edge < edges_end; ++edge) {
            VertexT neighbor = graph.column_indices[edge];
            
            if(source_path[neighbor] == -1) {
                // if unseen
                source_path[neighbor] = neighbor_dist;
                sigmas[neighbor] += sigmas[dequeued_node];
                if(search_depth < neighbor_dist) {
                    search_depth = neighbor_dist;
                }
                frontier.push_back(neighbor);
            } else {
                // if seen 
                if(source_path[neighbor] == source_path[dequeued_node] + 1) {
                    sigmas[neighbor] += sigmas[dequeued_node];
                }
            }
        }
    }
    search_depth++;
    
    for(int iter = search_depth - 2; iter > 0; --iter) {
        int cur_level = 0;
        for(int node = 0; node < graph.nodes; ++node) {
            if(source_path[node] == iter) {
                ++cur_level;
                
                int edges_begin = graph.row_offsets[node];
                int edges_end   = graph.row_offsets[node + 1];
                for(int edge = edges_begin; edge < edges_end; ++edge) {
                    VertexT neighbor = graph.column_indices[edge];
                    if(source_path[neighbor] == iter + 1) {
                         bc_values[node] += 
                            1.0f * sigmas[node] / sigmas[neighbor] * 
                            (1.0f + bc_values[neighbor]);
                    }
                }
            }
        }
    }
    
    for(int i =0; i < graph.nodes; ++i) {
        bc_values[i] *= 0.5f;
        std::cout << "i=" << i << " | bc_values[i]=" << bc_values[i] << std::endl;
    }

    for(int i =0; i < graph.nodes; ++i) {
        std::cout << "i=" << i << " | sigmas[i]=" << sigmas[i] << std::endl;
    }

    for(int i =0; i < graph.nodes; ++i) {
        std::cout << "i=" << i << " | source_path[i]=" << source_path[i] << std::endl;
    }
    
    cpu_timer.Stop();
    return cpu_timer.ElapsedMillis();
}

/**
 * @brief Validation of SSSP results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the distances
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
 * @param[in]  src           The source vertex
 * @param[in]  h_distances   Computed distances from the source to each vertex
 * @param[in]  h_preds       Computed predecessors for each vertex
 * @param[in]  ref_distances Reference distances from the source to each vertex
 * @param[in]  ref_preds     Reference predecessors for each vertex
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT,
    typename VertexT = typename GraphT::VertexT>
typename GraphT::SizeT Validate_Results(
            util::Parameters &parameters,
            GraphT           &graph,
            // TODO: add problem specific data for validation, e.g.:
            // - DONE
            VertexT   src,
            
            ValueT   *h_bc_values,
            ValueT   *h_sigmas,

            ValueT   *ref_bc_values = NULL,
            ValueT   *ref_sigmas = NULL,

            bool      verbose = true)
{
    typedef typename GraphT::SizeT   SizeT;
    // TODO: change to other representation, if not using CSR
    // - DONE
    typedef typename GraphT::CsrT    CsrT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");
    
    // Verify the result
    // TODO: result validation and display, e.g.:
    // if (ref_distances != NULL)
    // {
    //    for (VertexT v = 0; v < graph.nodes; v++)
    //    {
    //        if (!util::isValid(ref_distances[v]))
    //            ref_distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
    //    }
    //
    //    util::PrintMsg("Distance Validity: ", !quiet, false);
    //    SizeT errors_num = util::CompareResults(
    //        h_distances, ref_distances,
    //        graph.nodes, true, quiet);
    //    if (errors_num > 0)
    //    {
    //        util::PrintMsg(
    //            std::to_string(errors_num) + " errors occurred.", !quiet);
    //        num_errors += errors_num;
    //    }
    // }
    // else if (ref_distances == NULL)
    // {
    //    util::PrintMsg("Distance Validity: ", !quiet, false);
    //    SizeT errors_num = 0;
    //    for (VertexT v = 0; v < graph.nodes; v++)
    //    {
    //        ValueT v_distance = h_distances[v];
    //        if (!util::isValid(v_distance))
    //            continue;
    //        SizeT e_start = graph.CsrT::GetNeighborListOffset(v);
    //        SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
    //        SizeT e_end = e_start + num_neighbors;
    //        for (SizeT e = e_start; e < e_end; e++)
    //        {
    //            VertexT u = graph.CsrT::GetEdgeDest(e);
    //            ValueT u_distance = h_distances[u];
    //            ValueT e_value = graph.CsrT::edge_values[e];
    //            if (v_distance + e_value >= u_distance)
    //                continue;
    //            errors_num ++;
    //            if (errors_num > 1)
    //                continue;
    //
    //            util::PrintMsg("FAIL: v[" + std::to_string(v)
    //                + "] ("    + std::to_string(v_distance)
    //                + ") + e[" + std::to_string(e)
    //                + "] ("    + std::to_string(e_value)
    //                + ") < u[" + std::to_string(u)
    //                + "] ("    + std::to_string(u_distance) + ")", !quiet);
    //        }
    //    }
    //    if (errors_num > 0)
    //    {
    //        util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
    //        num_errors += errors_num;
    //    } else {
    //        util::PrintMsg("PASS", !quiet);
    //    }
    // }
    //
    // if (!quiet && verbose)
    // {
    //    // Display Solution
    //    util::PrintMsg("First 40 distances of the GPU result:");
    //    DisplaySolution(h_distances, graph.nodes);
    //    if (ref_distances != NULL)
    //    {
    //        util::PrintMsg("First 40 distances of the reference CPU result.");
    //        DisplaySolution(ref_distances, graph.nodes);
    //    }
    //    util::PrintMsg("");
    // }

    return num_errors;
}

} // namespace Template
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
