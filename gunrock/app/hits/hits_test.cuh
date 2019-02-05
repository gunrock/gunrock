// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * hits_test.cu
 *
 * @brief Test related functions for hits
 */

#pragma once

namespace gunrock {
namespace app {
namespace hits {

/******************************************************************************
 * HITS Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference hits ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph, 
    typename GraphT::ValueT *ref_hrank,
    typename GraphT::ValueT *ref_arank,
    typename GraphT::SizeT max_iter,
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // <TODO> 
    // implement CPU reference implementation
    // </TODO>

    // Temporary, just set to 0
    for(SizeT v = 0; v < graph.nodes; v++)
    {
        ref_hrank[v] = 0;
        ref_arank[v] = 0;

    }
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

// Structs and functions to rank hubs and authorities
template<typename RankPairT>
bool HITSCompare(
    const RankPairT &elem1,
    const RankPairT &elem2)
{
    return elem1.rank > elem2.rank;
}

template <typename GraphT>
struct RankPair
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  ValueT;

    VertexT        vertex_id;
    ValueT         rank;
};

template <typename GraphT>
struct RankList
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::SizeT   SizeT;
    
    typedef RankPair<GraphT> RankPairT;
    RankPairT *rankPairs;
    SizeT num_nodes;

    RankList(ValueT *ranks, SizeT num_nodes)
    {
        rankPairs = new RankPairT[num_nodes];

        this->num_nodes = num_nodes;

        for(SizeT v = 0; v < this->num_nodes; v++)
        {
            rankPairs[v].vertex_id = v;
            rankPairs[v].rank = ranks[v];
        }

        std::stable_sort(rankPairs, rankPairs+num_nodes, HITSCompare<RankPairT>);
    }

    ~RankList()
    {
        delete [] rankPairs;
        rankPairs = NULL;
    }

};



/**
 * @brief Displays the hits result
 *
 * @param[in] vertices Vertex Ids
 * @param[in] hrank HITS hub scores (sorted)
 * @param[in] arank HITS authority scores (sorted)
 * @param[in] num_vertices Number of vertices to display
 */
 template <typename GraphT>
 void DisplaySolution(
    typename GraphT::ValueT  *hrank,
    typename GraphT::ValueT  *arank,
    typename GraphT::SizeT    num_vertices
    )
 {
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::ValueT  ValueT;
    typedef typename GraphT::SizeT   SizeT;

    typedef RankList<GraphT> RankListT;
    RankListT hlist(hrank, num_vertices);
    RankListT alist(arank, num_vertices);

     // At most top 10 ranked vertices
     SizeT top = (num_vertices < 10) ? num_vertices : 10;

     util::PrintMsg("Top " + std::to_string(top)
         + " Ranks");

     for (SizeT i = 0; i < top; ++i)
     {
        util::PrintMsg("Vertex ID: " + std::to_string(hlist.rankPairs[i].vertex_id)
             + ", Hub Rank: " + std::to_string(hlist.rankPairs[i].rank));
        util::PrintMsg("Vertex ID: " + std::to_string(alist.rankPairs[i].vertex_id)
             + ", Authority Rank: " + std::to_string(alist.rankPairs[i].rank));
     }

     return;
 }



/**
 * @brief Validation of hits results
 * @tparam     GraphT        Type of the graph
 * @tparam     ValueT        Type of the values
 * @param[in]  parameters    Excution parameters
 * @param[in]  graph         Input graph
...
 * @param[in]  verbose       Whether to output detail comparsions
 * \return     GraphT::SizeT Number of errors
 */
template <typename GraphT>
typename GraphT::SizeT Validate_Results(
             util::Parameters &parameters,
             GraphT           &graph,
             typename GraphT::ValueT *h_hrank,
             typename GraphT::ValueT *h_arank,
             typename GraphT::ValueT *ref_hrank,
             typename GraphT::ValueT *ref_arank,
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::VertexT ValueT;
    typedef typename GraphT::SizeT   SizeT;

    typedef RankList<GraphT> RankListT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");
    bool quick = parameters.Get<bool>("quick");

    RankListT h_hlist(h_hrank, graph.nodes);
    RankListT h_alist(h_arank, graph.nodes);


    if (!quick)
    {
        // Add CPU references to RankList to sort
        RankListT ref_hlist(ref_hrank, graph.nodes);
        RankListT ref_alist(ref_arank, graph.nodes);

        for (SizeT v = 0; v < graph.nodes; v++)
        {
            if (ref_hlist.rankPairs[v].vertex_id != h_hlist.rankPairs[v].vertex_id) num_errors++;
            if (ref_alist.rankPairs[v].vertex_id != h_alist.rankPairs[v].vertex_id) num_errors++;
        }

        util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }



    return num_errors;
}

} // namespace hits
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
