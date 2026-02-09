#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace forman_ricci {

// =============================================================================
// Constants
// =============================================================================
constexpr float MIN_WEIGHT = 1e-6f;
constexpr float MAX_CURVATURE = 100000.0f;
constexpr float MIN_AREA = 1e-12f;
constexpr float STEP_SCALE = 1.1f;

// =============================================================================
// 1. param_t - Algorithm parameters
// =============================================================================
struct param_t {
  int n_iterations;
  param_t(int _n_iterations) : n_iterations(_n_iterations) {}
};

// =============================================================================
// 2. result_t - Output data structures
// =============================================================================
template <typename weight_t>
struct result_t {
  weight_t* edge_weights;  // Final edge weights after Ricci flow
  
  result_t(weight_t* _edge_weights) : edge_weights(_edge_weights) {}
};

// =============================================================================
// 3. problem_t - Graph + internal data structures
// =============================================================================
template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  param_type param;
  result_type result;

  // Internal data structures
  thrust::device_vector<vertex_t> undirected_src;
  thrust::device_vector<vertex_t> undirected_dst;
  edge_t n_undirected_edges;
  thrust::device_vector<weight_t> edge_curvature;
  weight_t max_curvature;
  weight_t total_weight;  

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  void init() override {
    auto g = this->get_graph();
    auto n_vertices = g.get_number_of_vertices();
    auto n_edges = g.get_number_of_edges();
    edge_curvature.resize(n_edges);

    // Build undirected edge list (u < v) from CSR
    // Copy CSR arrays to host (one-time cost)
    std::vector<edge_t> h_offsets(n_vertices + 1);
    std::vector<vertex_t> h_indices(n_edges);
    // cudaMemcpy(h_offsets.data(), g.get_row_offsets(),
    //           (n_vertices + 1) * sizeof(edge_t), cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_indices.data(), g.get_column_indices(),
    //           n_edges * sizeof(vertex_t), cudaMemcpyDeviceToHost);
    // thrust::copy(g.get_row_offsets(),
    //          g.get_row_offsets() + (n_vertices + 1),
    //          h_offsets.data());
    // thrust::copy(g.get_column_indices(),
    //             g.get_column_indices() + n_edges,
    //             h_indices.data());
    thrust::copy(thrust::device_pointer_cast(g.get_row_offsets()),
             thrust::device_pointer_cast(g.get_row_offsets()) + (n_vertices + 1),
             h_offsets.data());
    thrust::copy(thrust::device_pointer_cast(g.get_column_indices()),
                thrust::device_pointer_cast(g.get_column_indices()) + n_edges,
                h_indices.data());

    std::vector<vertex_t> src_vec, dst_vec;
    src_vec.reserve(n_edges / 2);
    dst_vec.reserve(n_edges / 2);

    for (vertex_t u = 0; u < n_vertices; u++) {
      for (edge_t e = h_offsets[u]; e < h_offsets[u + 1]; e++) {
        vertex_t v = h_indices[e];
        if (u < v) {
          src_vec.push_back(u);
          dst_vec.push_back(v);
        }
      }
    }

    n_undirected_edges = src_vec.size();
    undirected_src = src_vec;  // copies to device
    undirected_dst = dst_vec;
  }

  void reset() override {
    auto policy = this->context->get_context(0)->execution_policy();
    auto g = this->get_graph();
    auto n_edges = g.get_number_of_edges();

    // Initialize all edge weights to 1.0
    auto d_weights = thrust::device_pointer_cast(this->result.edge_weights);
    thrust::fill_n(policy, d_weights, n_edges, (weight_t)1.0);

    // Initialize curvatures to 0
    thrust::fill(policy, edge_curvature.begin(), edge_curvature.end(), (weight_t)0.0);
    
    max_curvature = 0.0;
    total_weight = 0.0;
  }
};

// =============================================================================
// 4. enactor_t - The computation logic
// =============================================================================
template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  // Track iterations
  int current_iteration = 0;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    auto n_undirected_edges = P->n_undirected_edges;  

    // Fill frontier with edge indices 0 to n_undirected_edges-1
    // Each "vertex" in the frontier is actually an edge index
    f->sequence((vertex_t)0, (vertex_t)n_undirected_edges, 
                context.get_context(0)->stream());
  }

  void loop(gcuda::multi_context_t& context) override {
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto n_vertices = G.get_number_of_vertices();
    auto n_total_edges = G.get_number_of_edges();
    auto edge_src = P->undirected_src.data().get();
    auto edge_dst = P->undirected_dst.data().get();
    auto n_undirected_edges = P->n_undirected_edges;

    auto edge_weights = P->result.edge_weights;
    auto edge_curvature = P->edge_curvature.data().get();

    auto policy = context.get_context(0)->execution_policy();

    // =========================================================================
    // Step 1: Compute Forman-Ricci curvature for each edge
    // =========================================================================
    auto compute_curvature = [G, edge_src, edge_dst, edge_weights, edge_curvature] 
        __host__ __device__(vertex_t const& edge_idx) -> bool {
      
      vertex_t v1 = edge_src[edge_idx];
      vertex_t v2 = edge_dst[edge_idx];

      // Find edge indices in CSR for (v1,v2) and (v2,v1)
      edge_t idx_v1_v2 = -1;
      edge_t idx_v2_v1 = -1;

      edge_t start_v1 = G.get_starting_edge(v1);
      edge_t end_v1 = start_v1 + G.get_number_of_neighbors(v1);
      edge_t start_v2 = G.get_starting_edge(v2);
      edge_t end_v2 = start_v2 + G.get_number_of_neighbors(v2);

      // Search v1's adjacency list for v2
      for (edge_t j = start_v1; j < end_v1; j++) {
        if (G.get_destination_vertex(j) == v2) {
          idx_v1_v2 = j;
          break;
        }
      }

      // Search v2's adjacency list for v1
      for (edge_t j = start_v2; j < end_v2; j++) {
        if (G.get_destination_vertex(j) == v1) {
          idx_v2_v1 = j;
          break;
        }
      }

      // edge_src/edge_dst are pre-filtered (u < v) from symmetric CSR,
      // so all edges are guaranteed to be found. Defensive check only.
      if (idx_v1_v2 < 0 || idx_v2_v1 < 0) {
        return true;  // Keep in frontier but skip
      }

      // Edge weight
      // weight_t w_e = max(edge_weights[idx_v1_v2], (weight_t)MIN_WEIGHT);
      weight_t w_e = fmax(edge_weights[idx_v1_v2], (weight_t)MIN_WEIGHT);

      // Vertex contribution
      weight_t sum_ve = 2.0f / w_e;

      // Triangle and parallel edge contributions
      weight_t triangle_contrib = 0.0f;
      weight_t sum_veeh = 0.0f;

      // Two-pointer merge to find triangles and parallel edges
      edge_t i1 = start_v1;
      edge_t i2 = start_v2;

      while (i1 < end_v1 && i2 < end_v2) {
        vertex_t n1 = G.get_destination_vertex(i1);
        vertex_t n2 = G.get_destination_vertex(i2);

        if (n1 == v2) { i1++; continue; }
        if (n2 == v1) { i2++; continue; }

        if (n1 == n2) {
          // Common neighbor - triangle
          weight_t w1 = fmax(edge_weights[i1], (weight_t)MIN_WEIGHT);
          weight_t w2 = fmax(edge_weights[i2], (weight_t)MIN_WEIGHT);

          // Heron's formula
          weight_t s = (w_e + w1 + w2) / 2.0f;
          weight_t area_sq = fabs(s * (s - w_e) * (s - w1) * (s - w2));
          weight_t w_tri = sqrt(fmax(area_sq, (weight_t)MIN_AREA));
          triangle_contrib += w_e / w_tri;

          i1++; i2++;
        } else if (n1 < n2) {
          // Parallel edge from v1
          weight_t w_ep = fmax(edge_weights[i1], (weight_t)MIN_WEIGHT);
          sum_veeh += 1.0f / sqrt(w_e * w_ep);
          i1++;
        } else {
          // Parallel edge from v2
          weight_t w_ep = fmax(edge_weights[i2], (weight_t)MIN_WEIGHT);
          sum_veeh += 1.0f / sqrt(w_e * w_ep);
          i2++;
        }
      }

      // Remaining neighbors of v1
      while (i1 < end_v1) {
        if (G.get_destination_vertex(i1) != v2) {
          weight_t w_ep = fmax(edge_weights[i1], (weight_t)MIN_WEIGHT);
          sum_veeh += 1.0f / sqrt(w_e * w_ep);
        }
        i1++;
      }

      // Remaining neighbors of v2
      while (i2 < end_v2) {
        if (G.get_destination_vertex(i2) != v1) {
          weight_t w_ep = fmax(edge_weights[i2], (weight_t)MIN_WEIGHT);
          sum_veeh += 1.0f / sqrt(w_e * w_ep);
        }
        i2++;
      }

      // Final curvature
      weight_t curvature = w_e * (triangle_contrib + sum_ve - sum_veeh);
      curvature = fmin((weight_t)MAX_CURVATURE, fmax((weight_t)-MAX_CURVATURE, curvature));

      edge_curvature[idx_v1_v2] = curvature;
      edge_curvature[idx_v2_v1] = curvature;

      return true;  // Keep in frontier for next iteration
    };

    // Execute curvature computation using filter (processes each edge)
    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, E, compute_curvature, context);

    // =========================================================================
    // Step 2: Find max absolute curvature
    // =========================================================================
    weight_t max_curv = thrust::transform_reduce(
        policy,
        P->edge_curvature.begin(),
        P->edge_curvature.end(),
        [] __host__ __device__(weight_t x) -> weight_t { return fabs(x); },
        (weight_t)0.0,
        thrust::maximum<weight_t>());

    P->max_curvature = max_curv;
    weight_t step_size = 1.0f / (STEP_SCALE * max_curv + 1e-10f);
    step_size = fmin(step_size, (weight_t)1.0);

    // =========================================================================
    // Step 3: Update weights
    // =========================================================================
    auto update_weights = [G, edge_src, edge_dst, edge_weights, edge_curvature, step_size]
        __host__ __device__(vertex_t const& edge_idx) -> bool {
      
      vertex_t v1 = edge_src[edge_idx];
      vertex_t v2 = edge_dst[edge_idx];

      edge_t idx_v1_v2 = -1;
      edge_t idx_v2_v1 = -1;

      edge_t start_v1 = G.get_starting_edge(v1);
      edge_t end_v1 = start_v1 + G.get_number_of_neighbors(v1);
      edge_t start_v2 = G.get_starting_edge(v2);
      edge_t end_v2 = start_v2 + G.get_number_of_neighbors(v2);

      for (edge_t j = start_v1; j < end_v1; j++) {
        if (G.get_destination_vertex(j) == v2) {
          idx_v1_v2 = j;
          break;
        }
      }

      for (edge_t j = start_v2; j < end_v2; j++) {
        if (G.get_destination_vertex(j) == v1) {
          idx_v2_v1 = j;
          break;
        }
      }

      if (idx_v1_v2 >= 0) {
        weight_t w_new = edge_weights[idx_v1_v2] * (1.0f - step_size * edge_curvature[idx_v1_v2]);
        w_new = fmax(w_new, (weight_t)MIN_WEIGHT);

        edge_weights[idx_v1_v2] = w_new;
        if (idx_v2_v1 >= 0) {
          edge_weights[idx_v2_v1] = w_new;
        }
      }

      return true;
    };

    operators::filter::execute<operators::filter_algorithm_t::predicated>(
        G, E, update_weights, context);

    // =========================================================================
    // Step 4: Normalize weights
    // =========================================================================
    auto d_weights = thrust::device_pointer_cast(edge_weights);
    weight_t total_weight = thrust::reduce(policy, d_weights, d_weights + n_total_edges,
                                           (weight_t)0.0, thrust::plus<weight_t>());

    P->total_weight = total_weight;
    weight_t scale = (weight_t)n_undirected_edges / (total_weight / 2.0f);

    thrust::transform(policy, d_weights, d_weights + n_total_edges, d_weights,
                      [scale] __host__ __device__(weight_t w) -> weight_t {
                        return w * scale;
                      });
    current_iteration++;
  }

  bool is_converged(gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    return current_iteration >= P->param.n_iterations;
  }

};  // struct enactor_t

// =============================================================================
// 5. run() - Entry point
// =============================================================================
template <typename graph_t>
float run(graph_t& G,
          int n_iterations,
          typename graph_t::weight_type* edge_weights,
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))) {

  using weight_t = typename graph_t::weight_type;

  param_t param(n_iterations);
  result_t<weight_t> result(edge_weights);

  using problem_type = problem_t<graph_t, param_t, result_t<weight_t>>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace forman_ricci
}  // namespace gunrock
