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
    typename GraphT::VertexT  src,
    bool                      quiet,
    bool                      mark_preds)
{

    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;
    typedef std::pair<VertexT, ValueT> PairT;
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
    for (int batch = 0; batch < num_batch; batch ++)
    {
        vector <Vertex> B_2;
        for (Vertex v = batch; v < batch + batch_size; v++)
        {
            B_2.push_back (v) ; // add node v to set B_2
        }//get B_2 done
        // construct set B_1
        vector <Vertex > B_1 ;
        for (int i = 0; i <B_2.size(); i++ ) 
        {
           // sample neighbours of each nodes in B_2
            Vertex node_v = B_2[i] ;
            SizeT num_neighbors = graph.GetNeighborListLength( node_v );
            SizeT e_start = graph.GetNeighborListOffset (node_v);
            for (int i2 =0; i2 < num_neigh1 ; i2 ++)
            {
                random_idx1 = rand() % num_neighbors ;
                sampled_node = graph.GetEdgeDest (e_start + random_idx);
                B_1.push_back (sampled_node);
            } // sample number of nodes for each node in B_2

        } //get B_1 done
        // construct set B_0
        vector <vector< Vertex>> B_0  ;
        for (int i = 0; i< B_1.size(); i++ ) 
        {
            Vertex node_v = B_1[i];
            SizeT = num_neighbors = graph.GetNeighborListLength( node_v );
            SizeT e_start = graph.GetNeighborListOffset (node_v);
            for (int i2 =0; i2 < num_neigh2 ; i2 ++)
            {
                random_idx1 = rand() % num_neighbors ;
                sampled_node = graph.GetEdgeDest (e_start + random_idx);
                B_0[i].push_back (sampled_node);
            }
 
        } // get B_0 done
        //distances[v] = util::PreDefinedValues<ValueT>::MaxValue;
        //if (mark_preds && preds != NULL)
            //preds[v] = util::PreDefinedValues<VertexT>::InvalidValue;
    }// get B^k done
                  
    // get the aggreate function
    // k =1 agg B0 nodes to B1 agg
    hv_k1_all = vector <vector<double>>;
    for (int i = 0 ; i < B_1.size() ; i++ ) {
        vector<double> agg_k1 = aggregate (B_0[i], features); //vector
        auto v = B_1[i];        
        vector <double> hv_k0 = features[v]; //vector
        vector <double> agg_k1_w1 = mm(agg_k1, w1_agg); // 64 to 128
        vector <double> hv_k0_w1 = mm (hv_k0, w1_hv); // 64 to 128  
        vector<double> hv_k1 = hv_k0_w1;
        hv_k1 = hv_k0_w1.insert (hv_k0_w1.end(), agg_k1_w1.begin(), agg_k1_w1.end() );
        // activation function
        hv_k1 = sigma (hv_k1);
        // normalize hv_k
        hv_k1 = L2_norm (hv_k1);     
        hv_k1_all.push_back  (hv_k1);
    }//for
    // agg B1 nodes to B2; 
    hu_k1_all = vector <vector<double>>;
    for (int i = 0; i<B_2.size(); i ++) 
    {   
        auto idx_start = i * num_neigh1;
        auto idx_end = (i+1) * num_neigh1;
        vector <double> agg_k2 = aggregate (B_1[idx_start: idx_end], features); 
        
        
             
    }//for 
    // K = 2
    hv_k2_all = vector <vector<double>>;
    for (int i = 0; i<B_2.size(); i ++) 
    {
        auto idx_start = i * num_neigh1;
        auto idx_end = (i+1) * num_neigh1;
        vector <double> agg_k2 = aggregate (B_1[idx_start: idx_end], hv_k1_all); 
        // get hv_k1
         
    }//for
}

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

    return num_errors;
}

} // namespace sage
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
