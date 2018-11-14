// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * knn_test.cu
 *
 * @brief Test related functions for knn
 */

#pragma once

namespace gunrock {
namespace app {
namespace knn {


/******************************************************************************
 * KNN Testing Routines
 *****************************************************************************/


/**
 * @brief Simple CPU-based reference knn ranking implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    int k, // number of nearest neighbor
    typename GraphT::VertexT point_x, // index of reference point
    typename GraphT::VertexT point_y, // index of reference point
    typename GraphT::ValueT *degrees,
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::ValueT ValueT;
    typedef typename GraphT::VertexT VertexT;
    
struct Point{
   VertexT x;
   VertexT y;
   VertexT dist;
   Point(){}
   Point(VertexT X, VertexT Y, VertexT Dist):x(X),y(Y),dist(Dist){}
};   

struct comp{
 inline bool operator()(const Point& p1, const Point& p2){
   return (p1.dist < p2.dist);
}
};
 
    Point* distance = (Point*)malloc(sizeof(Point)*graph.edges); 
    util::CpuTimer cpu_timer;
    cpu_timer.Start();
   
    // implement CPU reference implementation
    for (VertexT x = 0; x < graph.nodes; ++x) {
	for (SizeT i = graph.row_offsets[x]; i < graph.row_offsets[x + 1]; ++i) {
    	    VertexT y = graph.column_indices[i];		
	    VertexT dist = (x-point_x) * (x-point_x) + (y-point_y) * (y-point_y);
	    distance[i] = Point(x, y, dist);
	}
    }

    std::sort(distance, distance + graph.edges, comp());

    for (int i = 0; i < k; ++i){
	// k nearest points to (point_x, pointy)
	fprintf(stderr, "(%d, %d), ", distance[i].x, distance[i].y);
    }
    printf("\n");
    
    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of knn results
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
             // <TODO>
             typename GraphT::ValueT *h_degrees,
             typename GraphT::ValueT *ref_degrees,
             // </TODO>
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    // <TODO> result validation and display
    /*for(SizeT v = 0; v < graph.nodes; ++v) {
        printf("%d %d %d\n", v, h_degrees[v], ref_degrees[v]);
    }*/
    // </TODO>

    if(num_errors == 0) {
       util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }

    return num_errors;
}

} // namespace knn
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
