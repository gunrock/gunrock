// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * geo_spatial.cuh
 *
 * @brief Spatial helpers for geolocation app
 */

#pragma once

#include <cmath>
#include <algorithm>

namespace gunrock {
namespace app {
namespace geo {

/* This implementation is adapted from python geotagging app. */
#define PI 3.141592653589793

/**
 * @brief Compute the mean of all latitudues and
 *        and longitudes in a given set.
 */
template <typename ValueT, typename SizeT>
__host__ void mean(
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
__host__ void midpoint(
    ValueT p1_lat,
    ValueT p1_lon,
    ValueT p2_lat,
    ValueT p2_lon,
    ValueT * latitude,
    ValueT * longitude,
    SizeT v)
{
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
 * @brief Compute spatial median of a set of > 2 points.
 *
 *        Spatial Median;
 *        That is, given a set X find the point m s.t.
 *              sum([dist(x, m) for x in X])
 *
 *        is minimized. This is a robust estimator of
 *        the mode of the set.
 */
template <typename ValueT, typename SizeT>
__host__ void spatial_median(
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

    // Reinitialize these to ::problem
    // ValueT * D = new ValueT[length];
    // ValueT * Dinv = new ValueT[length];
    // ValueT * W = new ValueT[length];

    // <TODO> this will be a huge issue
    // if known locations for a node are
    // a lot! Will overflood the register
    // file.
    ValueT * D = new ValueT[length];
    ValueT * Dinv = new ValueT[length];
    ValueT * W = new ValueT[length];
    // </TODO>

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

    // Calculate the spatial median;
    while (true)
    {
        iter_ += 1;
        Dinvs = 0;
        T[0] = 0; T[1] = 0; // T.lat = 0, T.lon = 0;
        nonzeros = 0;

        for (SizeT k = 0; k < length; ++k)
        {
            D[k] = haversine(locations_lat[(v * offset) + k],
                             locations_lon[(v * offset) + k],
                             y[0], y[1]);

	
	    Dinv[k] = D[k] == 0 ? 0 : 1/D[k];
	    nonzeros = D[k] != 0 ? nonzeros + 1 : nonzeros;
	    Dinvs += Dinv[k];
	}

	for (SizeT k = 0; k < length; ++k)
	{
	    W[k] = Dinv[k]/Dinvs;
	    T[0] += W[k] * locations_lat[(v * offset) + k];
	    T[1] += W[k] * locations_lon[(v * offset) + k];
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
	    return;
	}

	y[0] = y1[0];
	y[1] = y1[1];
}
}

/**
 * @brief Compute "center" of a set of points.
 *
 *      For set X ->
 *        if points == 1; center = point;
 *        if points == 2; center = midpoint;
 *        if points > 2; center = spatial median;
 */
template <typename ValueT, typename SizeT>
__host__ void spatial_center(
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
        /* util::PrintMsg("Valid Locations [" + std::to_string(v) + "] : "
                        + std::to_string(length)
                        + " < " + std::to_string(latitude[v]) + " , "
                        + std::to_string(longitude[v]) + " > ", !quiet); */
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
                 /* util::PrintMsg("Valid Locations [" + std::to_string(v) + "] : "
                             + std::to_string(length)
                             + " < " + std::to_string(latitude[v]) + " , "
                             + std::to_string(longitude[v]) + " > ", !quiet);*/
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
     
             /* util::PrintMsg("Valid Locations [" + std::to_string(v) + "] : "
                             + std::to_string(length)
                             + " < " + std::to_string(latitude[v]) + " , "
                             + std::to_string(longitude[v]) + " > ", !quiet);*/
             return;
         }
     
}

} // namespace geo
} // namespace app
} // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
