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
template <typename GraphT, typename ArrayT>
double CPU_Reference(const GraphT &graph, ArrayT &latitude, ArrayT &longitude,
                     int geo_iter, int spatial_iter, bool geo_complete,
                     bool quiet) {
  typedef typename GraphT::SizeT SizeT;
  typedef typename GraphT::ValueT ValueT;
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::CsrT CsrT;

  util::Location target = util::HOST;

  SizeT nodes = graph.nodes;
  SizeT edges = graph.edges + 1;

  ValueT *Dinv = new ValueT[edges];

  int iterations = 0;

  // Number of nodes with known/predicted locations
  SizeT active = 0;
  bool Stop_Condition = false;

  util::CpuTimer cpu_timer;
  cpu_timer.Start();

  // implement CPU reference implementation
  while (!Stop_Condition) {
// Compute operator
#pragma omp parallel for
    for (VertexT v = 0; v < nodes; ++v) {
      SizeT offset = graph.GetNeighborListLength(v);
      if (!util::isValid(latitude[v]) && !util::isValid(longitude[v])) {
        ValueT neighbor_lat[2], neighbor_lon[2];

        SizeT start_edge = graph.CsrT::GetNeighborListOffset(v);
        SizeT num_neighbors = graph.CsrT::GetNeighborListLength(v);

        SizeT valid_neighbors = 0;

        for (SizeT e = start_edge; e < start_edge + num_neighbors; e++) {
          VertexT u = graph.CsrT::GetEdgeDest(e);
          if (util::isValid(latitude[u]) && util::isValid(longitude[u])) {
            neighbor_lat[valid_neighbors % 2] =
                latitude[u];  // last valid latitude
            neighbor_lon[valid_neighbors % 2] =
                longitude[u];  // last valid longitude
            valid_neighbors++;
          }
        }

        // If no locations found and no neighbors,
        // point at location (92.0, 182.0)
        if (valid_neighbors < 1)  // && offset == 0)
        {
        }

        // If one location found, point at that location
        if (valid_neighbors == 1) {
          latitude[v] = neighbor_lat[0];
          longitude[v] = neighbor_lon[0];
        }

        // If two locations found, compute a midpoint
        else if (valid_neighbors == 2) {
          midpoint(neighbor_lat[0], neighbor_lon[0], neighbor_lat[1],
                   neighbor_lon[1], latitude.GetPointer(target),
                   longitude.GetPointer(target), v);
        }

        // if locations more than 2, compute spatial
        // median.
        else {
          spatial_median(graph, valid_neighbors, latitude.GetPointer(target),
                         longitude.GetPointer(target), v, Dinv, quiet, target,
                         spatial_iter);
        }
      }
    }

    if (geo_complete) {
      active = 0;

      // Check all nodes with known location,
      // and increment active.
      for (SizeT v = 0; v < nodes; ++v) {
        if (util::isValid(latitude[v]) && util::isValid(longitude[v])) {
          active++;
        }
      }

      if (active == nodes) Stop_Condition = true;

    }

    else {
      if (iterations >= geo_iter) Stop_Condition = true;
    }

    iterations++;

  }  // -> while locations unknown.

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
template <typename GraphT, typename ArrayT>
typename GraphT::SizeT Validate_Results(
    util::Parameters &parameters, GraphT &graph,
    typename GraphT::ValueT *h_predicted_lat,
    typename GraphT::ValueT *h_predicted_lon, ArrayT &ref_predicted_lat,
    ArrayT &ref_predicted_lon, bool verbose = true) {
  typedef typename GraphT::VertexT VertexT;
  typedef typename GraphT::SizeT SizeT;

  SizeT num_errors = 0;
  bool quiet = parameters.Get<bool>("quiet");
  bool quick = parameters.Get<bool>("quick");

  if (!quick) {
    for (SizeT v = 0; v < graph.nodes; ++v) {
      printf("Node [ %lu ]: Predicted = < %f , %f > Reference = < %f , %f >\n",
             v, h_predicted_lat[v], h_predicted_lon[v], ref_predicted_lat[v],
             ref_predicted_lon[v]);
      if (!(h_predicted_lat[v] == ref_predicted_lat[v]) &&
          !(h_predicted_lon[v] == ref_predicted_lon[v])) {
        num_errors++;
      }
    }

    util::PrintMsg(std::to_string(num_errors) + " errors occurred.", !quiet);

  } else {
    printf("-------- NO VALIDATION --------\n");
  }

  return num_errors;
}

}  // namespace geo
}  // namespace app
}  // namespace gunrock

// Leave this at the end of the file
// Local Variables:
// mode:c++
// c-file-style: "NVIDIA"
// End:
