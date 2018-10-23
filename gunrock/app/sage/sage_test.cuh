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


template <typename ValueT, typename SizeT>
ValueT ** ReadMatrix (std::string filename, SizeT dim0, SizeT dim1)
{
    std::FILE* fin = fopen(filename.c_str(),"r");
    if (fin==NULL)
    {
        util::PrintMsg("Error in reading " + filename);
        return NULL;
    }

   ValueT **matrix = new ValueT*[dim0];
   for (SizeT i = 0; i < dim0; i++)
   {
        matrix[i] = new ValueT[dim1];
        for (SizeT j = 0; j < dim1; j++)
            fscanf(fin, "%f", matrix[i] + j);
   }
   fclose(fin);

    return matrix;
}


template <
    typename GraphT,
    typename ValueT = typename GraphT::ValueT>
double CPU_Reference(
    const GraphT   &graph,
          int       batch_size,
          int       num_neigh1,
          int       num_neigh2,
          ValueT  **features,
          ValueT  **W_f_1,
          ValueT  **W_a_1,
          ValueT  **W_f_2, 
          ValueT  **W_a_2,
          ValueT   *source_embedding,
          bool      quiet)
{
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
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

    util::PrintMsg("CPU_Reference entered", !quiet);
    //int num_batch = graph.nodes / batch_size ;
    //int off_site = graph.nodes - num_batch * batch_size ;
    // batch of nodes
    for (VertexT source_start = 0; source_start < graph.nodes ; source_start += batch_size)
    {
        int num_source = (source_start + batch_size <=graph.nodes ? batch_size: graph.nodes - source_start );
     
        util::PrintMsg("Processing sources [" + std::to_string(source_start) + ", "
            + std::to_string(source_start + num_source) + ")", !quiet); 
        for (VertexT source = source_start; source < source_start + num_source; source ++ )
        { 
            //store edges between sources and children 
            std::vector <SizeT> edges_source_child;
            float children_temp [256] = {0.0} ; // agg(h_B1^1)
            float source_temp [256] = {0.0};  // h_B2^1
            //float source_result [256] = {0.0}; // h_B2_2, result
            auto  source_result = source_embedding + source * 256;
            for (int i = 0; i < 256; i++)
                source_result[i] = 0.0;
            
            for (int i =0; i < num_neigh1 ; i++)
            {
                SizeT offset = rand() % num_neigh1; // YC: Bug
                SizeT pos = graph.GetNeighborListOffset(source) + offset;
                edges_source_child.push_back (pos);
            }// sample child (B1 nodes), save edge list. 

            // get each child's h_v^1
            for (int i = 0; i < num_neigh1 ; i ++)
            {
                SizeT pos = edges_source_child[i];
                VertexT child = graph.GetEdgeDest(pos); 
                float sums [64] = {0.0} ;

                // sample leaf node for each child
                for (int j =0; j < num_neigh2 ; j++)
                {
                    SizeT offset2 = rand() % num_neigh2; // YC: Bug
                    SizeT pos2 = graph.GetNeighborListOffset(child) + offset2;
                    VertexT leaf = graph.GetEdgeDest (pos2); 
                    for (int m = 0; m < 64 ; m ++) {
                        sums[m] += features[leaf][ m];
                    }
                    
                }// agg feaures for leaf nodes alg2 line 11 k = 1
                for (int m =0; m < 64 ; m++){
                    sums [m] = sums[m]/ num_neigh2;
                }// get mean  agg of leaf features.
                // get ebedding vector for child node (h_{B1}^{1}) alg2 line 12
                float child_temp[256] = {0.0};
                for (int idx_0 = 0; idx_0 < 128; idx_0++)
                {
                    for (int idx_1 =0; idx_1< 64; idx_1 ++)
                        child_temp[idx_0] += features[child][ idx_1] * W_f_1[idx_1][idx_0];
                } // got 1st half of h_B1^1

                for (int idx_0 = 128; idx_0 < 256; idx_0++)
                {   
                    for (int idx_1 =0; idx_1< 64; idx_1 ++)
                        child_temp[idx_0] += sums[idx_1] * W_a_1[idx_1][idx_0 - 128];
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
            float sums_child_feat [64] = {0.0} ; // agg(h_B1^0)
            for (int i = 0; i < num_neigh1 ; i ++)
            { 
                SizeT pos = edges_source_child[i];
                VertexT child = graph.GetEdgeDest(pos);
                for (int m = 0; m < 64; m++)
                {
                    sums_child_feat [m] += features[child][m];
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
                    source_temp[idx_0] += features[source][ idx_1] * W_f_1[idx_1][idx_0];
            } // got 1st half of h_B2^1

            for (int idx_0 = 128; idx_0 < 256; idx_0++)
            {
                for (int idx_1 =0; idx_1< 64; idx_1 ++)
                    source_temp[idx_0] += sums_child_feat[idx_1] * W_a_1[idx_1][idx_0 - 128];
            } // got 2nd half of h_B2^1 

            auto L2_source_temp = 0.0;
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_temp[idx_0] = source_temp[idx_0] > 0.0 ? source_temp[idx_0] : 0.0 ; //relu() 
                L2_source_temp += source_temp[idx_0] * source_temp [idx_0];
            } //finished relu
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_temp[idx_0] = source_temp[idx_0] /sqrt (L2_source_temp);
                //printf("source_temp,%f", source_temp[idx_0]);
            }//finished L-2 norm for source temp
            //////////////////////////////////////////////////////////////////////////////////////
            // get h_B2^2 k =2.           
            for (int idx_0 = 0; idx_0 < 128; idx_0++)
            {
                //printf ("source_r1_0:%f", source_result[idx_0] );
                for (int idx_1 =0; idx_1< 256; idx_1 ++)
                    source_result[idx_0] += source_temp [idx_1] * W_f_2[idx_1][idx_0];
                //printf ("source_r1:%f", source_result[idx_0] );
            } // got 1st half of h_B2^2

            for (int idx_0 = 128; idx_0 < 256; idx_0++)
            {
                //printf ("source_r2_0:%f", source_result[idx_0] );
                for (int idx_1 =0; idx_1< 256; idx_1 ++)
                    source_result[idx_0] += children_temp[idx_1] * W_a_2[idx_1][idx_0 - 128];
                //printf ("source_r2_1:%f", source_result[idx_0] );

            } // got 2nd half of h_B2^2
            auto L2_source_result = 0.0;
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_result[idx_0] = source_result[idx_0] > 0.0 ? source_result[idx_0] : 0.0 ; //relu() 
                L2_source_result += source_result[idx_0] * source_result [idx_0];
            } //finished relu
            for (int idx_0 =0; idx_0 < 256; idx_0 ++ )
            {
                source_result[idx_0] = source_result[idx_0] /sqrt (L2_source_result);
                //printf ("source_r:%f", source_result[idx_0] );
                //printf ("ch_t:%f", children_temp[idx_0]);
            }//finished L-2 norm for source result          
           
        } //for each source
        //printf ("node %d \n", source);
    } //for each batch
    util::PrintMsg("CPU_Reference exited", !quiet);
    cpu_timer.Stop();
    return cpu_timer.ElapsedMillis();  
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
             ValueT           *embed_result,
             int               result_column,
                     bool      verbose       = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    util::PrintMsg("Embedding validation: ", !quiet, false);
    // Verify the result
    for (SizeT v =0; v < graph.nodes; v++)
    {
        double L2_vec = 0.0;
        SizeT  offset = v * result_column;
        for (SizeT j = 0; j < result_column; j++)
        {
            L2_vec += embed_result[offset + j] * embed_result[offset + j] ;
        }  
        if (abs(L2_vec -1.0) > 0.000001)
        {
            if (num_errors == 0)
            {
                util::PrintMsg("FAIL. L2(embedding[" + std::to_string(v)
                    + "] = " + std::to_string(L2_vec) + ", should be 1", !quiet);
            }
           num_errors += 1;
        }
    }

    if (num_errors == 0)
        util::PrintMsg("PASS", !quiet);
    else 
        util::PrintMsg(
            std::to_string(num_errors) + " errors occurred.", !quiet);

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
