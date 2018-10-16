// ----------------------------------------------------------------
// Gunrock -- Fast and Efficient GPU Graph Library
// ----------------------------------------------------------------
// This source code is distributed under the terms of LICENSE.TXT
// in the root directory of this source distribution.
// ----------------------------------------------------------------

/**
 * @file
 * geo_d_spatial.cuh
 *
 * @brief Device Spatial helpers for geolocation app
 */

#pragma once

#include <cmath>
#include <algorithm>

namespace gunrock {
namespace app {
namespace geo {

/* This implementation is adapted from python geotagging app. */
#if 0
template <typename ValueT> 
__device__ const ValueT min (ValueT a, ValueT b) {
  return !(b < a ) ? a : b;     // or: return !comp(b,a)?a:b; for version (2)
}

template <typename ValueT>
__device__ const ValueT max (ValueT a, ValueT b) {
  return !(a < b ) ? b : a;     // or: return comp(a,b)?b:a; for version (2)
}
#endif

#define PI 3.141592653589793
// const double PI = 3.141592653589793;

template <typename ValueT>
__device__ __host__ ValueT radians(ValueT a){return a * PI/180;}

template <typename ValueT>
__device__ __host__ ValueT degrees(ValueT a){return a * 180/PI;}

/**
 * @brief Compute the mean of all latitudues and
 *        and longitudes in a given set.
 */
template <typename ValueT, typename SizeT, typename VertexT>
__device__ void mean(
    util::Array1D<SizeT, ValueT> latitude,
    util::Array1D<SizeT, ValueT> longitude,
    SizeT     length,
    ValueT  * mean,
    SizeT     offset,
    VertexT   v)
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

    mean[0] = a; // output latitude
    mean[1] = b; // output longitude

    return;
}

/**
 * @brief Compute the midpoint of two points on a sphere.
 */
template <typename ValueT, typename SizeT, typename VertexT>
__device__ void midpoint(
    ValueT p1_lat,
    ValueT p1_lon,
    ValueT p2_lat,
    ValueT p2_lon,
    util::Array1D<SizeT, ValueT> latitude,
    util::Array1D<SizeT, ValueT> longitude,
    VertexT v)
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
 * @brief (approximate) distance between two points on earth's
 * surface in kilometers.
 */
template <typename ValueT>
__device__ __host__ ValueT haversine(
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
 *        Spatial Median;
 *        That is, given a set X find the point m s.t.
 *              sum([dist(x, m) for x in X])
 *
 *        is minimized. This is a robust estimator of
 *        the mode of the set.
 */
template <typename ValueT, typename SizeT, typename VertexT>
__device__ void spatial_median(
    util::Array1D<SizeT, ValueT> locations_lat,
    util::Array1D<SizeT, ValueT> locations_lon,

    SizeT offset,
    SizeT length,

    util::Array1D<SizeT, ValueT> latitude,
    util::Array1D<SizeT, ValueT> longitude,

    VertexT v,

    util::Array1D<SizeT, ValueT> D,
    util::Array1D<SizeT, ValueT> Dinv,
    util::Array1D<SizeT, ValueT> W,

    bool quiet,
    ValueT eps = 1e-3,
    SizeT max_iter = 1000)
{

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
    mean (locations_lat, 
	  locations_lon,
          length, 
	  y, 
	  offset, 
	  v);

    // Calculate the spatial median;
    while (true)
    {
        iter_ += 1;
        Dinvs = 0;
        T[0] = 0; T[1] = 0; // T.lat = 0, T.lon = 0;
        nonzeros = 0;

        for (SizeT k = 0; k < length; ++k)
        {
            D[(v * offset) + k] = haversine(locations_lat[(v * offset) + k],
                             locations_lon[(v * offset) + k],
                             y[0], y[1]);

	
	    Dinv[(v * offset) + k] = D[(v * offset) + k] == 0 ? 0 : 1/D[(v * offset) + k];
	    nonzeros = D[(v * offset) + k] != 0 ? nonzeros + 1 : nonzeros;
	    Dinvs += Dinv[(v * offset) + k];
	}

	for (SizeT k = 0; k < length; ++k)
	{
	    W[(v * offset) + k] = Dinv[(v * offset) + k]/Dinvs;
	    T[0] += W[(v * offset) + k] * locations_lat[(v * offset) + k];
	    T[1] += W[(v * offset) + k] * locations_lon[(v * offset) + k];
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

	    y1[0] = fmaxf(0.0f, 1-rinv) * T[0]
	          + fminf(1.0f, rinv) * y[0]; // latitude
	    y1[1] = fmaxf(0.0f, 1-rinv) * T[1]
	          + fminf(1.0f, rinv) * y[1]; // longitude
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
template <typename ValueT, typename SizeT, typename VertexT>
__device__ void spatial_center(
    util::Array1D<SizeT, ValueT> locations_lat,
    util::Array1D<SizeT, ValueT> locations_lon,
 
    SizeT offset,
    SizeT length,

    util::Array1D<SizeT, ValueT> latitude,
    util::Array1D<SizeT, ValueT> longitude,

    VertexT v,

    util::Array1D<SizeT, ValueT> D,
    util::Array1D<SizeT, ValueT> Dinv,
    util::Array1D<SizeT, ValueT> W,

    bool quiet)
{
    // If no locations found and no neighbors,
    // point at location (92.0, 182.0)
    if (length < 1) // && offset == 0)
    {
#if 0
        latitude[v] = (ValueT) 92.0;
        longitude[v] = (ValueT) 182.0;
#endif
        return;
    }

    // If one location found, point at that location
    if (length == 1)
    {
        latitude[v] = locations_lat[(v * offset) + 0];
        longitude[v] = locations_lon[(v * offset) + 0];
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
			    D, Dinv, W,
                            quiet);
     
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
