/**
 * @file geo.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief
 * @version 0.1
 * @date 2021-02-12
 *
 * @copyright Copyright (c) 2021
 *
 */

#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace geo {

struct coordinates_t {
  float latitude;
  float longitude;
};

// Value of PI
#define PI 3.141592653589793

/**
 * @brief degrees -> radians
 *	      radians -> degrees
 */
__device__ __host__ __forceinline__ float radians(float a) {
  return a * PI / 180;
}

__device__ __host__ __forceinline__ float degrees(float a) {
  return a * 180 / PI;
}

/**
 * @brief Compute the mean of all latitudues and longitudes in a given set.
 */
template <typename graph_t, typename vertex_t = typename graph_t::vertex_type>
__device__ __host__ coordinates_t mean(graph_t const& G,
                                       coordinates_t const* coordinates,
                                       std::size_t const& length,
                                       vertex_t const& v) {
  // Calculate mean;
  float a = 0;
  float b = 0;

  auto start_edge = G.get_starting_edge(v);
  auto num_neighbors = G.get_number_of_neighbors(v);

  for (auto e = start_edge; e < start_edge + num_neighbors; e++) {
    vertex_t u = G.get_destination_vertex(e);
    if (gunrock::util::limits::is_valid(coordinates[u].latitude) &&
        gunrock::util::limits::is_valid(coordinates[u].longitude)) {
      // Accumulate the valid latitude and longitude values
      a += coordinates[u].latitude;
      b += coordinates[u].longitude;
    }
  }

  a /= length;
  b /= length;

  // mean latitude and longitude
  return {a, b};
}

/**
 * @brief Compute the midpoint of two points on a sphere.
 */
template <typename vertex_t>
__device__ __host__ coordinates_t midpoint(coordinates_t p1,
                                           coordinates_t p2,
                                           vertex_t const& v) {
  // Convert to radians
  p1.latitude = radians(p1.latitude);
  p1.longitude = radians(p1.longitude);
  p2.latitude = radians(p2.latitude);
  p2.longitude = radians(p2.longitude);

  float bx, by;
  bx = cos(p2.latitude) * cos(p2.longitude - p1.longitude);
  by = cos(p2.latitude) * sin(p2.longitude - p1.longitude);

  coordinates_t mid;
  mid.latitude =
      atan2(sin(p1.latitude) + sin(p2.latitude),
            sqrt((cos(p1.latitude) + bx) * (cos(p1.latitude) + bx) + by * by));

  mid.longitude = p1.longitude + atan2(by, cos(p1.latitude) + bx);

  // Convert back to degrees and return the coordinate.
  mid.latitude = degrees(mid.latitude);
  mid.longitude = degrees(mid.longitude);

  return mid;
}

/**
 * @brief (approximate) distance between two points on earth's
 * surface in kilometers.
 */
__device__ __host__ float haversine(coordinates_t n,
                                    coordinates_t mean,
                                    float radius = 6371) {
  // Convert degrees to radians
  n.latitude = radians(n.latitude);
  n.longitude = radians(n.longitude);
  mean.latitude = radians(mean.latitude);
  mean.longitude = radians(mean.longitude);

  float lat = mean.latitude - n.latitude;
  float lon = mean.longitude - n.longitude;

  float a = pow(sin(lat / 2), 2) +
            cos(n.latitude) * cos(mean.latitude) * pow(sin(lon / 2), 2);

  float c = 2 * asin(sqrt(a));

  // haversine distance in km
  return radius * c;
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
template <typename graph_t, typename vertex_t = typename graph_t::vertex_type>
__device__ __host__ void spatial_median(graph_t& G,
                                        std::size_t length,
                                        coordinates_t* coordinates,
                                        vertex_t v,
                                        float* Dinv,
                                        int max_iter = 1000,
                                        float eps = 1e-3) {
  float r, rinv;
  float Dinvs = 0;

  std::size_t iter_ = 0;
  std::size_t nonzeros = 0;
  std::size_t num_zeros = 0;

  // Can be done cleaner, but storing lat and long.
  coordinates_t R, tmp, T;
  coordinates_t y, y1;

  auto start_edge = G.get_starting_edge(v);
  auto num_neighbors = G.get_number_of_neighbors(v);

  // Calculate mean of all <latitude, longitude>
  // for all possible locations of v;
  y = mean(G, coordinates, length, v);

  // Calculate the spatial median;
  while (true) {
    iter_ += 1;
    Dinvs = 0;
    T.latitude = 0;
    T.longitude = 0;  // T.lat = 0, T.lon = 0;
    nonzeros = 0;

    for (auto e = start_edge; e < start_edge + num_neighbors; e++) {
      vertex_t u = G.get_destination_vertex(e);
      if (gunrock::util::limits::is_valid(coordinates[u].latitude) &&
          gunrock::util::limits::is_valid(coordinates[u].longitude)) {
        // Get the haversine distance between the latitude and longitude of
        // valid neighbor
        auto Dist = haversine(coordinates[u], y);
        Dinv[e] = Dist == 0 ? 0 : 1 / Dist;
        nonzeros = Dist != 0 ? nonzeros + 1 : nonzeros;
        Dinvs += Dinv[e];
      }
    }

    std::size_t len = 0;
    for (auto e = start_edge; e < start_edge + num_neighbors; e++) {
      // W[] array, Dinv[e] / Dinvs
      vertex_t u = G.get_destination_vertex(e);
      if (gunrock::util::limits::is_valid(coordinates[u].latitude) &&
          gunrock::util::limits::is_valid(coordinates[u].longitude)) {
        T.latitude += (Dinv[e] / Dinvs) * coordinates[u].latitude;
        T.longitude += (Dinv[e] / Dinvs) * coordinates[u].longitude;
        len++;
      }
    }

    num_zeros = len - nonzeros;
    if (num_zeros == 0) {
      y1.latitude = T.latitude;
      y1.longitude = T.longitude;
    }

    else if (num_zeros == len) {
      // Valid location found
      coordinates[v].latitude = y.latitude;
      coordinates[v].longitude = y.longitude;
      return;
    }

    else {
      R.latitude = (T.latitude - y.latitude) * Dinvs;
      R.longitude = (T.longitude - y.longitude) * Dinvs;
      r = sqrt(R.latitude * R.latitude + R.longitude * R.longitude);

      // Was rinv = (r == 0) ?: (num_zeros / r);
      // https://gcc.gnu.org/onlinedocs/gcc/Conditionals.html
      // ... I hate myself too.
      rinv = (r == 0) ? 0 : (num_zeros / r);

      y1.latitude = max(0.0f, 1 - rinv) * T.latitude +
                    min(1.0f, rinv) * y.latitude;  // latitude
      y1.longitude = max(0.0f, 1 - rinv) * T.longitude +
                     min(1.0f, rinv) * y.longitude;  // longitude
    }

    tmp.latitude = y.latitude - y1.latitude;
    tmp.longitude = y.longitude - y1.longitude;

    if ((sqrt(tmp.latitude * tmp.latitude + tmp.longitude * tmp.longitude)) <
            eps ||
        (iter_ > max_iter)) {
      // Valid location found
      coordinates[v].latitude = y1.latitude;
      coordinates[v].longitude = y1.longitude;
      return;
    }

    y.latitude = y1.latitude;
    y.longitude = y1.longitude;
  }  // -> spatial_median while() loop
}

struct param_t {
  unsigned int total_iterations;
  unsigned int spatial_iterations;
  param_t(unsigned int _total_iterations, unsigned int _spatial_iterations)
      : total_iterations(_total_iterations),
        spatial_iterations(_spatial_iterations) {}
};

struct result_t {
  coordinates_t* coordinates;
  result_t(coordinates_t* _coordinates) : coordinates(_coordinates) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<float> inv_haversine_distance;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  void init() override {
    auto g = this->get_graph();
    auto n_edges = g.get_number_of_edges();
    inv_haversine_distance.resize(n_edges);
  }

  void reset() override {
    /// @todo reset the coordinates and inv_haversine_distance arrays.
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context,
            enactor_properties_t _properties)
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto coordinates = P->result.coordinates;
    auto spatial_iterations = P->param.spatial_iterations;
    auto inv_haversine_distance = P->inv_haversine_distance.data().get();

    /**
     * @brief Compute "center" of a set of points.
     *
     *      For set X ->
     *        if points == 1; center = point;
     *        if points == 2; center = midpoint;
     *        if points > 2; center = spatial median;
     */
    auto spatial_center_op = [=] __device__(vertex_t const& v) -> void {
      if (gunrock::util::limits::is_valid(coordinates[v].latitude) &&
          gunrock::util::limits::is_valid(coordinates[v].longitude))
        return;

      // if no predicted location, and neighbor locations exists
      // Custom spatial center median calculation for geolocation
      // start median calculation -->
      auto start_edge = G.get_starting_edge(v);
      auto num_neighbors = G.get_number_of_neighbors(v);

      auto valid_neighbors = 0;
      coordinates_t neighbors[2];  // for length <=2 use registers

      for (auto e = start_edge; e < start_edge + num_neighbors; e++) {
        auto u = G.get_destination_vertex(e);
        if (gunrock::util::limits::is_valid(coordinates[u].latitude) &&
            gunrock::util::limits::is_valid(coordinates[u].longitude)) {
          neighbors[valid_neighbors % 2].latitude =
              coordinates[u].latitude;  // last valid latitude
          neighbors[valid_neighbors % 2].longitude =
              coordinates[u].longitude;  // last valid longitude
          valid_neighbors++;
        }
      }

      // If one location found, point at that location
      if (valid_neighbors == 1) {
        coordinates_t only_neighbor;
        if (gunrock::util::limits::is_valid(neighbors[0].latitude) &&
            gunrock::util::limits::is_valid(neighbors[0].longitude)) {
          only_neighbor.latitude = neighbors[0].latitude;
          only_neighbor.longitude = neighbors[0].longitude;
        } else {
          only_neighbor.latitude = neighbors[1].latitude;
          only_neighbor.longitude = neighbors[1].longitude;
        }
        coordinates[v].latitude = only_neighbor.latitude;
        coordinates[v].longitude = only_neighbor.longitude;
        return;
      }

      // If two locations found, compute a midpoint
      else if (valid_neighbors == 2) {
        coordinates[v] = midpoint(neighbors[0], neighbors[1], v);
        return;
      }

      // if locations more than 2, compute spatial median.
      else if (valid_neighbors > 2) {
        spatial_median(G, valid_neighbors, coordinates, v,
                       inv_haversine_distance, spatial_iterations);
      }

      // if no valid locations are found
      else {
        coordinates[v].latitude = gunrock::numeric_limits<float>::invalid();
        coordinates[v].longitude = gunrock::numeric_limits<float>::invalid();
      }

      // <-- end median calculation.
    };

    /*!
     * For each vertex, run the spatial center operation.
     * @todo We can possibly do better with a filter operator instead.
     */
    operators::parallel_for::execute<operators::parallel_for_each_t::vertex>(
        G,                  // graph
        spatial_center_op,  // lambda function
        context             // context
    );
  }

  bool is_converged(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto iteration = E->iteration;
    auto total_iterations = P->param.total_iterations;

    if (iteration == total_iterations)
      return true;
    else
      return false;
  }
  // </user-defined>
};  // struct enactor_t

template <typename graph_t>
float run(graph_t& G,
          coordinates_t* coordinates,                    // Input/Output
          const unsigned int total_iterations,           // Parameter
          const unsigned int spatial_iterations = 1000,  // Parameter
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  // <user-defined>
  using param_type = param_t;
  using result_type = result_t;

  param_type param(total_iterations, spatial_iterations);
  result_type result(coordinates);
  // </user-defined>

  // <boiler-plate>
  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;

  enactor_type enactor(&problem, context, props);
  return enactor.enact();
  // </boiler-plate>
}

}  // namespace geo
}  // namespace gunrock