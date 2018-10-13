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

#include<cmath>
#include<algorithm>


namespace gunrock {
namespace app {
// <DONE> change namespace
namespace geo {
// </DONE>


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

/* This implementation is adapted from python geotagging app. */
const double PI = 3.141592653589793;

template <typename ValueT>
ValueT radians(ValueT a){return a * PI/180;}

template <typename ValueT>
ValueT degrees(ValueT a){return a * 180/PI;}

/**
 * @brief Compute the mean of all latitidues and
 * 	  and longitudes in a given set.
 */
template <typename ValueT, typename SizeT>
void mean(
    ValueT *latitude,
    ValueT *longitude,
    SizeT   length,
    ValueT *y,
    SizeT   offset,
    SizeT   v) 
{
    // Calculate mean;
    ValueT a = 0;
    ValueT b = 0;

    for(SizeT k = 0; k < length; k++)
    {
	a += latitude[(v * offset) + k];
	b += longitude[(v * offset) + k];
    }

    a /= length;
    b /= length;

    y[0] = a; // output latitude
    y[1] = b; // output longitude

    return;
}

/**
 * @brief Compute the midpoint of two points on a sphere.
 */
template <typename ValueT, typename SizeT>
void midpoint(
    ValueT p1_lat,
    ValueT p1_lon,
    ValueT p2_lat,
    ValueT p2_lon,
    ValueT * latitude,
    ValueT * longitude,
    SizeT v)
{
    printf("p1 of [ %u ] : < %f , %f >.\n", v, p1_lat, p1_lon);
    printf("p2 of [ %u ] : < %f , %f >.\n", v, p2_lat, p2_lon);

    // Convert to radians
    p1_lat = radians(p1_lat);
    p1_lon = radians(p1_lon);
    p2_lat = radians(p2_lat);
    p2_lon = radians(p2_lon);

    ValueT bx, by;
    bx = cos(p2_lat) * cos(p2_lon - p1_lon);
    by = cos(p2_lat) * sin(p2_lon - p1_lon);

    ValueT lat, lon;
    lat = atan2(sin(p1_lat) + sin(p2_lat), \
		sqrt((cos(p1_lat) + bx) * (cos(p1_lat) \
		+ bx) + by*by));

    lon = p1_lon + atan2(by, cos(p1_lat) + bx);

    latitude[v] = degrees(lat);
    longitude[v] = degrees(lon);

    return;
}

/**
 * @brief (approximate) distance between two points on earth's 
 * surface in kilometers.
 */
template <typename ValueT>
ValueT haversine(
    ValueT n_latitude,
    ValueT n_longitude,
    ValueT mean_latitude,
    ValueT mean_longitude,
    ValueT radius = 6371)
{
    ValueT lat, lon;

    // Convert degrees to radians
    n_latitude = radians(n_latitude);
    n_longitude = radians(n_longitude);
    mean_latitude = radians(mean_latitude);
    mean_longitude = radians(mean_longitude);

    lat = mean_latitude - n_latitude;
    lon = mean_longitude - n_longitude;

    ValueT a = pow(sin(lat/2),2) + cos(n_latitude) *
		cos(mean_latitude) * pow(sin(lon/2),2);

    ValueT c = 2 * asin(sqrt(a));

    // haversine distance
    ValueT km =  radius * c;
    return km;
}

/**
 * @brief Compute spatial median of a set of > 2 points.
 *
 *	  Spatial Median;
 *	  That is, given a set X find the point m s.t.
 *		sum([dist(x, m) for x in X])
 *
 *	  is minimized. This is a robust estimator of
 *	  the mode of the set.
 */
template <typename ValueT, typename SizeT>
void spatial_median(
    ValueT *locations_lat,
    ValueT *locations_lon,
    SizeT offset,
    SizeT length,
    ValueT *latitude,
    ValueT *longitude,
    SizeT v,
    bool quiet,
    ValueT eps = 1e-3,
    SizeT max_iter = 1000)
{

    ValueT * D = new ValueT[length];
    ValueT * Dinv = new ValueT[length];
    ValueT * W = new ValueT[length];

    ValueT r, rinv;
    ValueT Dinvs = 0;

    SizeT iter_ = 0;
    SizeT nonzeros = 0;
    SizeT num_zeros = 0;
    
    // Can be done cleaner, but storing lat and long.
    ValueT R[2], tmp[2], out[2], T[2];
    ValueT y[2], y1[2];

    // Calculate mean of all <latitude, longitude>
    // for all possible locations of v;
    mean (locations_lat, locations_lon,
	    length, y, offset, v); 

    util::PrintMsg("Mean of all neighbor locations: " + std::to_string(length)
                        + " < " + std::to_string(y[0]) + " , "
                        + std::to_string(y[1]) + " > ", !quiet);

    // Calculate the spatial median;
    while (true)
    {
	iter_ += 1;
	Dinvs = 0;
	T[0] = 0; T[1] = 0; // T.lat = 0, T.lon = 0;
	nonzeros = 0;

	for (SizeT k = 0; k < length; ++k)
	{
	    D[k] = haversine(locations_lat[(v + offset) + k],
			     locations_lon[(v + offset) + k], 
			     y[0], y[1]);

//	    util::PrintMsg("Haversine distance: " + std::to_string(D[k]), 
//				!quiet);

	    Dinv[k] = D[k] == 0 ? 0 : 1/D[k];
	    nonzeros = D[k] != 0 ? nonzeros + 1 : nonzeros;
	    Dinvs += Dinv[k];
	}

	for (SizeT k = 0; k < length; ++k)
	{
	    W[k] = Dinv[k]/Dinvs;
	    T[0] += W[k] * locations_lat[(v + offset) + k];
	    T[1] += W[k] * locations_lon[(v + offset) + k];
	}

	num_zeros = length - nonzeros;
	if (num_zeros == 0)
	{
	    y1[0] = T[0];
	    y1[1] = T[1];
	}

	else if (num_zeros == length)
	{
	    latitude[v] = y[0];
	    longitude[v] = y[1];
	    util::PrintMsg("Prediction found [" + std::to_string(v) + " ] : "
                        + " < " + std::to_string(y[0]) + " , "
                        + std::to_string(y[1]) + " > ", !quiet);

	    return;
	}

	else
	{
	    R[0] = (T[0] - y[0]) * Dinvs;
	    R[1] = (T[1] - y[1]) * Dinvs;
	    r = sqrt(R[0] * R[0] + R[1] * R[1]);
	    rinv = r == 0 ? : num_zeros / r;
	    
	    y1[0] = std::max((ValueT) 0.0, (ValueT) 1-rinv) * T[0] 
		  + std::min((ValueT) 1.0, (ValueT) rinv) * y[0]; // latitude
            y1[1] = std::max((ValueT) 0.0, (ValueT) 1-rinv) * T[1] 
		  + std::min((ValueT) 1.0, (ValueT) rinv) * y[1]; // longitude
	}

	tmp[0] = y[0] - y1[0];
	tmp[1] = y[1] - y1[1];

	if((sqrt(tmp[0] * tmp[0] + tmp[1] * tmp[1])) < eps || (iter_ > max_iter))
	{
	    latitude[v] = y1[0];
	    longitude[v] = y1[1];
	    util::PrintMsg("Prediction found [" + std::to_string(v) + " ] : "
                        + " < " + std::to_string(y1[0]) + " , "
                        + std::to_string(y1[1]) + " > ", !quiet);
	    return;
	}

	y[0] = y1[0];
	y[1] = y1[1];	
    }
}


/**
 * @brief Compute "center" of a set of points.
 *
 *	For set X ->
 *	  if points == 1; center = point;
 *	  if points == 2; center = midpoint;
 *	  if points > 2; center = spatial median;
 */
template <typename ValueT, typename SizeT>
void spatial_center(
    ValueT * locations_lat,
    ValueT * locations_lon,
    SizeT offset,
    SizeT length,
    ValueT * latitude,
    ValueT * longitude,
    SizeT v,
    bool quiet)
{
    // If no locations found and no neighbors,
    // point at location (92.0, 182.0)
    if (length < 1) // && offset == 0) 
    {
#if 0
	latitude[v] = (ValueT) 92.0;
	longitude[v] = (ValueT) 182.0;
	util::PrintMsg("Valid Locations [" + std::to_string(v) + "] : " 
			+ std::to_string(length)
                        + " < " + std::to_string(latitude[v]) + " , "
                        + std::to_string(longitude[v]) + " > ", !quiet);
#endif
	return;
    }

    // If one location found, point at that location
    if (length == 1)
    {
	latitude[v] = locations_lat[(v * offset) + 0];
	longitude[v] = locations_lon[(v * offset) + 0];
	util::PrintMsg("Valid Locations [" + std::to_string(v) + "] : " 
			+ std::to_string(length)
                        + " < " + std::to_string(latitude[v]) + " , "
                        + std::to_string(longitude[v]) + " > ", !quiet);
	return;
    }

    // If two locations found, compute a midpoint
    else if (length == 2)
    {
	midpoint(locations_lat[(v * offset) + 0],
		 locations_lon[(v * offset) + 0],
		 locations_lat[(v * offset) + 1],
		 locations_lon[(v * offset) + 1],
		 latitude,
		 longitude,
		 v);
	util::PrintMsg("Valid Locations [" + std::to_string(v) + "] : " 
			+ std::to_string(length)
                        + " < " + std::to_string(latitude[v]) + " , "
                        + std::to_string(longitude[v]) + " > ", !quiet);
	return;
    }

    // if locations more than 2, compute spatial
    // median.
    else 
    {
	spatial_median(locations_lat, 
		       locations_lon,
		       offset, 
		       length,
		       latitude, 
		       longitude,
		       v, 
		       quiet);	

	util::PrintMsg("Valid Locations [" + std::to_string(v) + "] : "
			+ std::to_string(length)
                        + " < " + std::to_string(latitude[v]) + " , "
                        + std::to_string(longitude[v]) + " > ", !quiet);
	return;
    }

}


template <typename GraphT>
double CPU_Reference(
    const GraphT &graph,
    // <DONE> add problem specific inputs and outputs 
    typename GraphT::ValueT *predicted_lat,
    typename GraphT::ValueT *predicted_lon,
    // </DONE>
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

	util::PrintMsg("Current Predicted Locations: " 
			+ std::to_string(active), !quiet);

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

    // <DONE> result validation and display
    for(SizeT v = 0; v < graph.nodes; ++v) {
        printf("Node [ %d ]: Predicted = < %f , %f > Reference = < %f , %f >\n", v, 
		h_predicted_lat[v], h_predicted_lon[v], 
		ref_predicted_lat[v], ref_predicted_lon[v]);
    }
    // </DONE>

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
