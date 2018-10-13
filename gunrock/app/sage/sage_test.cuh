// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * sage_test.cu
 *
 * @brief Test related functions for SSSP
 */

#pragma once

#ifdef BOOST_FOUND
    // Boost includes for CPU Dijkstra SSSP reference algorithms
    #include <boost/config.hpp>
    #include <boost/graph/graph_traits.hpp>
    #include <boost/graph/adjacency_list.hpp>
    #include <boost/graph/dijkstra_shortest_paths.hpp>
    #include <boost/property_map/property_map.hpp>
#else
    #include <queue>
    #include <vector>
    #include <utility>
#endif

namespace gunrock {
namespace app {
namespace sage {

/******************************************************************************
 * Housekeeping Routines
 ******************************************************************************/

/**
 * @brief Displays the SAGE result.
 * @tparam T Type of values to display
 * @tparam SizeT Type of size counters
 * @param[in] preds Search depth from the source for each node.
 * @param[in] num_nodes Number of nodes in the graph.
 */
template<typename T, typename SizeT>
void DisplaySolution(T *array, SizeT length)
{
    if (length > 40)
        length = 40;

    util::PrintMsg("[", true, false);
    for (SizeT i = 0; i < length; ++i)
    {
        util::PrintMsg(std::to_string(i) + ":"
            + std::to_string(array[i]) + " ", true, false);
    }
    util::PrintMsg("]");
}

/******************************************************************************
 * SAGE Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference SSSP implementations
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
    typename ValueT = typename GraphT::ValueT>



double CPU_Reference(
    const    GraphT          &graph,
    //                 ValueT  *distances,
    int                batch_size,
    int                num_neigh1,
    int                num_neigh2,
    int **             features,
    int **             W_1_f,
    int **             W_1_a,
    int **             W_2_f, 
    int **             W_2_a,
    //typename GraphT::VertexT  src,
    bool                      quiet,
    bool                      mark_preds)
{

    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    //typedef std::pair<VertexT, ValueT> PairT;
    //struct GreaterT
    //{
    //    bool operator()(const PairT& lhs, const PairT& rhs)
    //    {
    //        return lhs.second > rhs.second;
    //    }
    //};
    //typedef std::priority_queue<PairT, std::vector<PairT>, GreaterT> PqT;

    int num_batch = graph.nodes / batch_size ;
    int off_site = graph.nodes - num_batch * batch_size ;
    // batch of nodes
    for (Vertex source_start = 0; source_start < graph.nodes ; source_start += batch_size)
    {
        int num_source = (source_start + batch_size <=graph.nodes ? batch_size: graph.nodes - source_start );
        
        for (Vertex source = source_start; source < source_start + num_sourse; source ++ )
        { 
            //store edges between sources and children 
            vector <SizeT> edges_source_child;
            auto children_temp [256] = {0.0} ; // agg(h_B1^1)
            auto source_temp [256] = {0.0};  // h_B2^1
            auto source_result [256] = {0.0}; // h_B2_2, result
            
            for (int i =0; i < num_neigh1 ; i++)
            {
                SizeT offset = rand() % num_neigh1;
                SizeT pos = graph.GetNeighborListOffset(source) + offset;
                edges_source_child.push_back (pos);
            }// sample child (B1 nodes), save edge list. 

            // get each child's h_v^1
            for (int i = 0; i < num_neigh1 ; i ++)
            {
                SizeT pos = edges_source_child[i];
                Vertex child = graph.GetEdgeDest(pos); 
                auto sums [64] = {0.0} ;

                // sample leaf node for each child
                for (int j =0; j < num_neigh2 ; j++)
                {
                    SizeT offset2 = rand() % num_neigh2;
                    SizeT pos2 = graph.GetNeighborListOffset(child) + offset2;
                    Vertex leaf = graph.GetEdgeDest (pos2); 
                    for (int m = 0; m < 64 ; m ++) {
                        sums[m] += features[leaf, m];
                    }
                    
                }// agg feaures for leaf nodes alg2 line 11 k = 1
                for (int m =0; m < 64 ; m++){
                    sums [m] = sums[m]/ num_neigh2;
                }// get mean  agg of leaf features.
                // get ebedding vector for child node (h_{B1}^{1}) alg2 line 12
                auto child_temp[256] = {0.0};
                for (int idx_0 = 0; idx_0 < 128; idx_0++)
                {
                    for (int idx_1 =0; idx_1< 64; idx_1 ++)
                        child_temp[idx_0] += features[child, idx_1] * W_f_1[idx_1,idx_0];
                } // got 1st half of h_B1^1

                for (int idx_0 = 128; idx_0 < 256; idx_0++)
                {   
                    for (int idx_1 =0; idx_1< 64; idx_1 ++)
                        child_temp[idx_0] += sums[idx_1] * W_a_1[idx_1,idx_0 - 128];
                } // got 2nd half of h_B!^1 

                // activation and L-2 normalize 
                auto L2_child_temp = 0.0;
                for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
                {
                    child_temp[idx_0] = child_temp[idx_0] > 0.0 ? child_temp[idx_0] : 0.0 ; //relu() 
                    L2_child_temp += child_temp[idx_0] * child_temp [idx_0];
                } //finished relu
                for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
                {
                    child_temp[idx_0] = child_temp[idx_0] /sqrt (L2_child_temp);
                }//finished L-2 norm, got h_B1^1, algo2 line13

                // add the h_B1^1 to children_temp, also agg it
                for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
                {
                    children_temp[idx_0] += child_temp[idx_0] /num_neigh1;
                }//finished agg (h_B1^1)
            }//for each child in B1, got h_B1^1

            //////////////////////////////////////////////////////////////////////////////////////
            //get h_B2^1, k =1 ; this time, child is like leaf, and source is like child
            auto sums_child_feat [64] = {0.0} ; // agg(h_B1^0)
            for (int i = 0; i < num_neigh1 ; i ++)
            { 
                SizeT pos = edges_source_child[i];
                Vertex child = graph.GetEdgeDest(pos);
                for (int m = 0; m < 64; m++)
                {
                    sums_child_feat [m] += features[child,m];
                }
                 
            }// for each child
            for (int m = 0 ; m < 64; m++)
            {
               sums_child_feat[m] = sums_child_feat[m]/ num_neigh1;
            } // got agg(h_B1^0)
 
            // get ebedding vector for child node (h_{B2}^{1}) alg2 line 12            
            for (int idx_0 = 0; idx_0 < 128; idx_0++)
            {
                for (int idx_1 =0; idx_1< 64; idx_1 ++)
                    source_temp[idx_0] += features[source, idx_1] * W_f_1[idx_1,idx_0];
            } // got 1st half of h_B2^1

            for (int idx_0 = 128; idx_0 < 256; idx_0++)
            {
                for (int idx_1 =0; idx_1< 64; idx_1 ++)
                    source_temp[idx_0] += sums_child_feat[idx_1] * W_a_1[idx_1,idx_0 - 128];
            } // got 2nd half of h_B2^1 

            auto L2_source_temp = 0.0;
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_temp[idx_0] = source_temp[idx_0] > 0.0 ? source_temp[idx_0] : 0.0 ; //relu() 
                L2_source_temp += source_temp[idx_0] * soruce_temp [idx_0];
            } //finished relu
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_temp[idx_0] = source_temp[idx_0] /sqrt (L2_source_temp);
            }//finished L-2 norm for source temp
            //////////////////////////////////////////////////////////////////////////////////////
            // get h_B2^2 k =2.           
            for (int idx_0 = 0; idx_0 < 128; idx_0++)
            {
                for (int idx_1 =0; idx_1< 256; idx_1 ++)
                    source_result[idx_0] += source_temp [idx_1] * W_f_2[idx_1,idx_0];
            } // got 1st half of h_B2^2

            for (int idx_0 = 128; idx_0 < 256; idx_0++)
            {
                for (int idx_1 =0; idx_1< 256; idx_1 ++)
                    source_result[idx_0] += children_temp[idx_1] * W_a_2[idx_1,idx_0 - 128];
            } // got 2nd half of h_B2^2
            auto L2_source_result = 0.0;
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_result[idx_0] = source_result[idx_0] > 0.0 ? source_result[idx_0] : 0.0 ; //relu() 
                L2_source_result += source_result[idx_0] * soruce_result [idx_0];
            } //finished relu
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_result[idx_0] = source_result[idx_0] /sqrt (L2_source_result);
            }//finished L-2 norm for source result 
           
        } //for each source
    } //for each batch
    return 0;  
} //cpu reference 

/**
 * @brief Validation of SAGE results
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
    typename ValueT = typename GraphT::ValueT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
    typename GraphT::VertexT   src,
                     ValueT   *h_distances,
    typename GraphT::VertexT  *h_preds,
                     ValueT   *ref_distances = NULL,
    typename GraphT::VertexT  *ref_preds     = NULL,
                     bool      verbose       = true)
{
/*
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef typename GraphT::CsrT    CsrT;

    SizeT num_errors = 0;
    //bool quick = parameters.Get<bool>("quick");
    bool quiet = parameters.Get<bool>("quiet");
    bool mark_pred = parameters.Get<bool>("mark-pred");

    // Verify the result
    if (ref_distances != NULL)
    {
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            if (!util::isValid(ref_distances[v]))
                ref_distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
        }

        util::PrintMsg("Distance Validity: ", !quiet, false);
        SizeT errors_num = util::CompareResults(
            h_distances, ref_distances,
            graph.nodes, true, quiet);
        if (errors_num > 0)
        {
            util::PrintMsg(
                std::to_string(errors_num) + " errors occurred.", !quiet);
            num_errors += errors_num;
        }
    }
    else if (ref_distances == NULL)
    {
        util::PrintMsg("Distance Validity: ", !quiet, false);
        SizeT errors_num = 0;
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            ValueT v_distance = h_distances[v];
            if (!util::isValid(v_distance))
                continue;
            SizeT e_start = graph.CsrT::GetNeighborListOffset(v);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);
            SizeT e_end = e_start + num_neighbors;
            for (SizeT e = e_start; e < e_end; e++)
            {
                VertexT u = graph.CsrT::GetEdgeDest(e);
                ValueT u_distance = h_distances[u];
                ValueT e_value = graph.CsrT::edge_values[e];
                if (v_distance + e_value >= u_distance)
                    continue;
                errors_num ++;
                if (errors_num > 1)
                    continue;

                util::PrintMsg("FAIL: v[" + std::to_string(v)
                    + "] ("    + std::to_string(v_distance)
                    + ") + e[" + std::to_string(e)
                    + "] ("    + std::to_string(e_value)
                    + ") < u[" + std::to_string(u)
                    + "] ("    + std::to_string(u_distance) + ")", !quiet);
            }
        }
        if (errors_num > 0)
        {
            util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
            num_errors += errors_num;
        } else {
            util::PrintMsg("PASS", !quiet);
        }
    }

    if (!quiet && verbose)
    {
        // Display Solution
        util::PrintMsg("First 40 distances of the GPU result:");
        DisplaySolution(h_distances, graph.nodes);
        if (ref_distances != NULL)
        {
            util::PrintMsg("First 40 distances of the reference CPU result.");
            DisplaySolution(ref_distances, graph.nodes);
        }
        util::PrintMsg("");
    }

    if (mark_pred)
    {
        util::PrintMsg("Predecessors Validity: ", !quiet, false);
        SizeT errors_num = 0;
        for (VertexT v = 0; v < graph.nodes; v++)
        {
            VertexT pred          = h_preds[v];
            if (!util::isValid(pred) || v == src)
                continue;
            ValueT  v_distance    = h_distances[v];
            if (v_distance == util::PreDefinedValues<ValueT>::MaxValue)
                continue;
            ValueT  pred_distance = h_distances[pred];
            bool edge_found = false;
            SizeT edge_start = graph.CsrT::GetNeighborListOffset(pred);
            SizeT num_neighbors = graph.CsrT::GetNeighborListLength(pred);

            for (SizeT e = edge_start; e < edge_start + num_neighbors; e++)
            {
                if (v == graph.CsrT::GetEdgeDest(e) &&
                    std::abs((pred_distance + graph.CsrT::edge_values[e]
                    - v_distance) * 1.0) < 1e-6)
                {
                    edge_found = true;
                    break;
                }
            }
            if (edge_found)
                continue;
            errors_num ++;
            if (errors_num > 1)
                continue;

            util::PrintMsg("FAIL: [" + std::to_string(pred)
                + "] ("    + std::to_string(pred_distance)
                + ") -> [" + std::to_string(v)
                + "] ("    + std::to_string(v_distance)
                + ") can't find the corresponding edge.", !quiet);
        }
        if (errors_num > 0)
        {
            util::PrintMsg(std::to_string(errors_num) + " errors occurred.", !quiet);
            num_errors += errors_num;
        } else {
            util::PrintMsg("PASS", !quiet);
        }
    }

    if (!quiet && mark_pred && verbose)
    {
        util::PrintMsg("First 40 preds of the GPU result:");
        DisplaySolution(h_preds, graph.nodes);
        if (ref_preds != NULL)
        {
            util::PrintMsg("First 40 preds of the reference CPU result "
                "(could be different because the paths are not unique):");
            DisplaySolution(ref_preds, graph.nodes);
        }
        util::PrintMsg("");
    }
*/
    return 0;
}

} // namespace sage
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
