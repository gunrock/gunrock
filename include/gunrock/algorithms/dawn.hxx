#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
// The utilization of advance within DAWN is effective.
namespace dawn_bfs {

template <typename vertex_t>
struct param_t {
  vertex_t single_source;
  param_t(vertex_t _single_source) : single_source(_single_source) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* distances;
  vertex_t* predecessors;  /// @todo: implement this.
  result_t(vertex_t* _distances, vertex_t* _predecessors)
      : distances(_distances), predecessors(_predecessors) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  void init() override {}

  void reset() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    thrust::fill(thrust::device, d_distances + 0, d_distances + n_vertices, 0);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;

  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();
    auto policy = this->context->get_context(0)->execution_policy();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;

    auto iteration = this->iteration + 1;

    auto sovm = [distances, single_source, iteration] __host__ __device__(
                    vertex_t const& source,    // ... source
                    vertex_t const& neighbor,  // neighbor
                    edge_t const& edge,        // edge
                    weight_t const& weight     // weight (tuple).
                    ) -> bool {
      auto old_distance = thread::load(&distances[neighbor]);
      if ((!old_distance) && (neighbor != single_source)) {
        // if (!old_distance) {
        distances[neighbor] = iteration;
        return true;
      }
      return false;
    };

    auto remove_invalids =
        [] __host__ __device__(vertex_t const& vertex) -> bool {
      // Returning true here means that we keep all the valid vertices.
      // Internally, filter will automatically remove invalids and will
      // never pass them to this lambda function.
      return true;
    };

    // Execute advance operator on the provided lambda
    operators::advance::execute<operators::load_balance_t::block_mapped>(
        G, E, sovm, context);
  }

};  // struct enactor_t

/**
 * @brief Run DAWN algorithm on a given graph, G, starting from
 * the source node, single_source. The resulting distances are stored in the
 * distances pointer. All data must be allocated by the user, on the device
 * (GPU) and passed in to this function.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param single_source A vertex in the graph (integral type).
 * @param distances Pointer to the distances array of size number of vertices.
 * @param predecessors Pointer to the predecessors array of size number of
 * vertices. (optional, wip)
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t>
float run(graph_t& G,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::vertex_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t<vertex_t>;
  using result_type = result_t<vertex_t>;

  param_type param(single_source);
  result_type result(distances, predecessors);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace dawn_bfs

// The utilization of parallel_for within DAWN can't work.
namespace dawn_bfs_visited {

template <typename vertex_t, typename graph_t, typename csr_t>
struct param_t {
  vertex_t single_source;
  graph_t& G;
  csr_t& A;
  param_t(vertex_t _single_source, graph_t& _G, csr_t& _A)
      : single_source(_single_source), G(_G), A(_A) {}
};

template <typename vertex_t>
struct result_t {
  vertex_t* distances;
  vertex_t* predecessors;  /// @todo: implement this.
  result_t(vertex_t* _distances, vertex_t* _predecessors)
      : distances(_distances), predecessors(_predecessors) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  param_type param;
  result_type result;

  problem_t(graph_t& G,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(G, _context),
        param(_param),
        result(_result) {}

  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;

  thrust::device_vector<vertex_t> visited;  /// @todo not used.
  void init() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    visited.resize(n_vertices, 0);
  }

  // We have inspected and verified the accurate ingress of the data at this
  // location.
  void reset() override {
    auto n_vertices = this->get_graph().get_number_of_vertices();
    auto single_source = this->param.single_source;

    auto& csr = this->param.A;
    auto d_distances = thrust::device_pointer_cast(this->result.distances);
    auto start = csr.row_offsets[single_source];
    auto end = csr.row_offsets[single_source + 1];
    for (vertex_t j = start; j < end; ++j) {
      auto idx = csr.column_indices[j];
      thrust::fill(thrust::device, d_distances + idx, d_distances + idx + 1, 1);
    }
    thrust::copy(thrust::device, d_distances, d_distances + n_vertices,
                 visited.begin());
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<gcuda::multi_context_t> _context)
      : gunrock::enactor_t<problem_t>(_problem, _context) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;
  using frontier_t = typename enactor_t<problem_t>::frontier_t;
  using csr_t = graph::
      graph_csr_t<memory::memory_space_t::device, vertex_t, edge_t, weight_t>;

  // We are uncertain whether the invocation at this juncture pertains to an
  // active frontier, and despite an exhaustive project-wide search, we have
  // been unable to locate an instance of data injection at this juncture.
  // Consequently, we have restructured a lambda function within the loop to
  // update the active frontier.
  void prepare_frontier(frontier_t* f,
                        gcuda::multi_context_t& context) override {
    auto P = this->get_problem();
    f->push_back(P->param.single_source);
  }

  void loop(gcuda::multi_context_t& context) override {
    // Data slice
    auto E = this->get_enactor();
    auto P = this->get_problem();
    auto G = P->get_graph();

    auto policy = this->context->get_context(0)->execution_policy();
    auto n_vertices = G.get_number_of_vertices();

    auto single_source = P->param.single_source;
    auto distances = P->result.distances;
    auto iteration = this->iteration + 1;
    auto visited = P->visited.data().get();

    // We aim to utilize this function to update the active frontier, where the
    // "visited" vector retains the updated values from the previous iteration,
    // serving to determine the active frontier based on these updated values.
    auto sovm_remove_invalids = [visited, single_source] __host__ __device__(
                                    vertex_t const& row) -> bool {
      if (row != single_source)
        return visited[row];
      return false;
    };

    // Regardless of whether data is input using CSR or G, it seems that SOVM is
    // ineffective. Upon inspecting the output, we have observed that the
    // "distances" array retains its initialized values. We believe that our
    // understanding of data input in the lambda function may be inadequate, or
    // alternatively, our comprehension of the parallel_for parallel function
    // may be insufficient.
    auto sovm =
        [distances, visited, iteration, single_source, G] __host__ __device__(
            vertex_t const& row) -> void {
      auto start = G.get_starting_edge(row);
      auto end = G.get_number_of_neighbors(row) + start;
      for (edge_t i = start; i < end; ++i) {
        auto idx = G.get_destination_vertex(i);
        if ((!distances[idx]) & (idx != single_source)) {
          distances[idx] = iteration;
          visited[idx] = 1;
        }
      }
    };
    //  auto& A = P->param.A;
    //  auto& row_offsets = A.row_offsets;
    //  auto& column_indices = A.column_indices;
    //  edge_t* row_off = row_offsets.data().get();
    //  vertex_t* col_ind = column_indices.data().get();
    //  auto sovm = [distances, iteration, single_source, row_off, col_ind]
    //  __host__
    //              __device__(vertex_t const& row) -> void {
    //    auto start = thread::load(&row_off[row]);
    //    auto end = thread::load(&row_off[row + 1]);
    //    for (edge_t i = start; i < end; ++i) {
    //      auto idx = thread::load(&col_ind[i]);
    //      if ((!distances[idx]) & (idx != single_source)) {
    //        distances[idx] = iteration;
    //        visited[idx] = 1;
    //      }
    //    }
    //  };

    // We intend to utilize the filter to update the active frontier. According
    // to the comments in the filter.hxx file, invoking it in this manner should
    // automatically generate a new frontier from the original frontier
    // and perform a pointer swap. It is possible that our misunderstanding of
    // the filter functionality is impeding the updating of the frontier,
    // consequently causing the function to malfunction.
    operators::filter::execute<operators::filter_algorithm_t::bypass>(
        G, E, sovm_remove_invalids, context);
    thrust::fill(policy, visited, visited + n_vertices, 0);
    operators::parallel_for::execute<operators::parallel_for_each_t::element>(
        *(E->get_input_frontier()), sovm, context);
  }

  // We aim to terminate the loop when the active frontier is empty. We have
  // modeled our implementation based on other files utilizing this function.
  virtual bool is_converged(gcuda::multi_context_t& context) {
    auto frontier = this->get_enactor()->get_input_frontier();
    return (!frontier->is_empty());
  }

};  // struct enactor_t

/**
 * @brief Run DAWN algorithm on a given graph, G, starting from
 * the source node, single_source. The resulting distances are stored in the
 * distances pointer. All data must be allocated by the user, on the device
 * (GPU) and passed in to this function.
 *
 * @tparam graph_t Graph type.
 * @param G Graph object.
 * @param A CSR object.
 * @param single_source A vertex in the graph (integral type).
 * @param distances Pointer to the distances array of size number of vertices.
 * @param predecessors Pointer to the predecessors array of size number of
 * vertices. (optional, wip)
 * @param context Device context.
 * @return float Time taken to run the algorithm.
 */
template <typename graph_t, typename csr_t>
float run(graph_t& G,
          csr_t& A,
          typename graph_t::vertex_type& single_source,  // Parameter
          typename graph_t::vertex_type* distances,      // Output
          typename graph_t::vertex_type* predecessors,   // Output
          std::shared_ptr<gcuda::multi_context_t> context =
              std::shared_ptr<gcuda::multi_context_t>(
                  new gcuda::multi_context_t(0))  // Context
) {
  using vertex_t = typename graph_t::vertex_type;
  using param_type = param_t<vertex_t, graph_t, csr_t>;
  using result_type = result_t<vertex_t>;

  param_type param(single_source, G, A);
  result_type result(distances, predecessors);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(G, param, result, context);
  problem.init();
  problem.reset();

  enactor_type enactor(&problem, context);
  return enactor.enact();
}

}  // namespace dawn_bfs_visited

namespace dawn_sssp {}  // namespace dawn_sssp
}  // namespace gunrock