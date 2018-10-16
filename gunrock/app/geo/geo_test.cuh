// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * geo_test.cu
 *
 * @brief Test related functions for geo
 */

#pragma once

#include <gunrock/app/geo/geo_spatial.cuh>

namespace gunrock {
namespace app {
namespace geo {


/******************************************************************************
 * Geolocation Testing Routines
 *****************************************************************************/

/**
 * @brief Simple CPU-based reference geolocation implementations
 * @tparam      GraphT        Type of the graph
 * @tparam      ValueT        Type of the values
 * @param[in]   graph         Input graph
...
 * @param[in]   quiet         Whether to print out anything to stdout
 */
template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    typename GraphT::ValueT *predicted_lat,
    typename GraphT::ValueT *predicted_lon,
    bool quiet)
{
    typedef typename GraphT::SizeT SizeT;
    typedef typename GraphT::ValueT ValueT;
    typedef typename GraphT::VertexT VertexT;

    SizeT nodes = graph.nodes;

    // Arrays to store neighbor locations
    ValueT * locations_lat = new ValueT[graph.nodes * graph.nodes];
    ValueT * locations_lon = new ValueT[graph.nodes * graph.nodes];

    // Number of valid locations for specific vertex
    SizeT * valid_locations = new SizeT[graph.nodes];

    // Number of nodes with known/predicted locations
    SizeT active = 0;
    bool Stop_Condition = false;

    util::CpuTimer cpu_timer;
    cpu_timer.Start();
    
    // <TODO> 
    // implement CPU reference implementation
    while (!Stop_Condition) 
    {
	// Gather operator
	// #pragma omp parallel
	for (SizeT v = 0; v < nodes; ++v) 
	{
	    
            SizeT start_e = graph.GetNeighborListOffset(v);
            SizeT degree  = graph.GetNeighborListLength(v);
	
	    SizeT i = 0;		

	    // if location not known:
	    if (!util::isValid(predicted_lat[v]) &&
		!util::isValid(predicted_lon[v])) 
	    {
		// #pragma omp parallel
	        for (SizeT k = 0; k < degree; k++) 
		{
		    SizeT e   = start_e + k;
		    VertexT u = graph.GetEdgeDest(e);

		    if (util::isValid(predicted_lat[u]) &&
			util::isValid(predicted_lon[u])) 
		    {
			locations_lat[(v * degree) + i] = predicted_lat[u];
			locations_lon[(v * degree) + i] = predicted_lon[u];
			// printf("gather : < %lf , %lf >\n", predicted_lat[u], predicted_lon[u]);
			i++;
		    }

		}
		
		valid_locations[v] = i;
	    }
        } // end: gather (for)
  
	// Compute operator 
	// #pragma omp parallel
	for (SizeT v = 0; v < nodes; ++v) 
	{
	    SizeT offset  = graph.GetNeighborListLength(v);
	    if (!util::isValid(predicted_lat[v]) &&
                !util::isValid(predicted_lon[v]))
	    {

//		util::PrintMsg("--- Spatial Center Operator --- ", !quiet);
		spatial_center( locations_lat, 	      // list of neigh latitudes 
				locations_lon, 	      // list of neigh longitudes
				offset,		      // degree of vertex v	
				valid_locations[v],   // number of valid locations
				predicted_lat,	      // output latitude
				predicted_lon,	      // output longitude
				v,		      // active vertex
			        quiet);	
	    }
	}
	
	active = 0;

	// Check all nodes with known location,
	// and increment active.
	for (SizeT v = 0; v < nodes; ++v) 
	{
	    if (util::isValid(predicted_lat[v]) && 
		util::isValid(predicted_lon[v])) 
	    {
		active++;
	    }
	}

	if(active == nodes) 
	    Stop_Condition = true; 

	// util::PrintMsg("Current Predicted Locations: " 
	// 		+ std::to_string(active), !quiet);

    } // -> while locations unknown.
    // </TODO>

    cpu_timer.Stop();
    float elapsed = cpu_timer.ElapsedMillis();
    return elapsed;
}

/**
 * @brief Validation of geolocation results
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
             typename GraphT::ValueT *h_predicted_lat,
             typename GraphT::ValueT *h_predicted_lon,
             typename GraphT::ValueT *ref_predicted_lat,
             typename GraphT::ValueT *ref_predicted_lon,
             bool verbose = true)
{
    typedef typename GraphT::VertexT VertexT;
    typedef typename GraphT::SizeT   SizeT;

    SizeT num_errors = 0;
    bool quiet = parameters.Get<bool>("quiet");

    for(SizeT v = 0; v < graph.nodes; ++v) {
        printf("Node [ %d ]: Predicted = < %f , %f > Reference = < %f , %f >\n", v, 
		h_predicted_lat[v], h_predicted_lon[v], 
		ref_predicted_lat[v], ref_predicted_lon[v]);
    }

    if(num_errors == 0) {
       util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);
    }

    return num_errors;
}

} // namespace geo
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
