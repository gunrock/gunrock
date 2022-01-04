/**
 * @file spgemm.hxx
 * @author Muhammad Osama (mosama@ucdavis.edu)
 * @brief Sparse-Matrix-Matrix multiplication.
 * @version 0.1
 * @date 2022-01-04
 *
 * @copyright Copyright (c) 2022
 *
 */

#pragma once

#include <gunrock/algorithms/algorithms.hxx>

namespace gunrock {
namespace spgemm {

template <typename a_graph_t, typename b_graph_t>
struct param_t {
  a_graph_t& A;
  b_graph_t& B;
  param_t(a_graph_t& _A, b_graph_t& _B) : A(_A), B(_B) {}
};

template <typename graph_t>
struct result_t {
  graph_t& C;
  result_t(graph_t& _C) : C(_C) {}
};

template <typename graph_t, typename param_type, typename result_type>
struct problem_t : gunrock::problem_t<graph_t> {
  using edge_t = typename graph_t::edge_type;

  param_type param;
  result_type result;

  problem_t(graph_t& A,
            param_type& _param,
            result_type& _result,
            std::shared_ptr<cuda::multi_context_t> _context)
      : gunrock::problem_t<graph_t>(A, _context),
        param(_param),
        result(_result) {}

  thrust::device_vector<std::size_t> d_nz_per_row;

  void init() override {
    auto A = this->get_graph();

    // Allocate memory for row-offsets of C = m + 1.
    auto C_row_offsets = result.C.get_row_offsets();
    memory::allocate(C_row_offsets,
                     (A.get_number_of_vertices() + 1) * sizeof(edge_t));

    d_nz_per_row.resize(A.get_number_of_vertices() + 1);
  }

  void reset() override {
    auto policy = this->context->get_context(0)->execution_policy();
    auto g = this->get_graph();
    thrust::fill(policy, d_nz_per_row.begin(), d_nz_per_row.end(), 0);
  }
};

template <typename problem_t>
struct enactor_t : gunrock::enactor_t<problem_t> {
  enactor_t(problem_t* _problem,
            std::shared_ptr<cuda::multi_context_t> _context,
            enactor_properties_t _properties = enactor_properties_t())
      : gunrock::enactor_t<problem_t>(_problem, _context, _properties) {}

  using vertex_t = typename problem_t::vertex_t;
  using edge_t = typename problem_t::edge_t;
  using weight_t = typename problem_t::weight_t;

  void loop(cuda::multi_context_t& context) override {}

  void push(cuda::multi_context_t& context) {
    auto E = this->get_enactor();
    auto P = this->get_problem();

    auto A = P->param.A;
    auto B = P->param.B;
    auto C = P->result.C;

    auto C_row_offsets = C.get_row_offsets();
    auto C_col_indices = C.get_col_indices();
    auto C_values = C.get_values();
    auto C_nnz_per_row = P->d_nz_per_row.data().get();

    /// Step 1. Count number of nonzeros per row of C.
    auto count_nonzeros =
        [=] __host__ __device__(
            vertex_t const& source,    // ... source (row index)
            vertex_t const& neighbor,  // neighbor (column index)
            edge_t const& edge,        // edge (row ↦ column)
            weight_t const& weight     // weight (nonzero).
            ) -> bool {
      // Compute number of nonzeros of the sparse-matrix C for each row.
      math::atomic::add(&C_nnz_per_row[source],
                        B.get_number_of_neighbors(neighbor));
      return false;  // ignored.
    };

    // Perform advance on the above lambda-op
    operators::advance::execute<
        operators::load_balance_t::block_mapped,
        operators::advance_direction_t::forward,  // direction (backward for
                                                  // transpose)
        operators::advance_io_type_t::graph,      // entire graph as input
        operators::advance_io_type_t::none>(      // no output frontier needed
        A, E, count_nonzeros, context);

    /// Step 2. Calculate total number of nonzeros in the sparse-matrix C.
    auto policy = this->context->get_context(0)->execution_policy();
    edge_t C_nnz =
        thrust::reduce(policy, P->d_nz_per_row.begin(), P->d_nz_per_row.end(),
                       (edge_t)0, thrust::plus<edge_t>());

    /// Step 3. Allocate memory for C's values and column indices.
    memory::allocate(C_col_indices, C_nnz * sizeof(vertex_t));
    memory::allocate(C_values, C_nnz * sizeof(weight_t));

    /// Step 4. Calculate C's row-offsets.
    thrust::transform_exclusive_scan(policy, P->d_nz_per_row.begin(),
                                     P->d_nz_per_row.end(), C_row_offsets,
                                     edge_t(0), thrust::plus<edge_t>());

    /// Step 5. Calculate C's column indices and values.
    auto spgemm = [=] __host__ __device__(
                      vertex_t const& source,    // ... source (row index)
                      vertex_t const& neighbor,  // neighbor (column index)
                      edge_t const& edge,        // edge (row ↦ column)
                      weight_t const& weight     // weight (nonzero).
                      ) -> bool {
      auto starting_edge = B.get_starting_edge(neighbor);
      auto total_edges = B.get_number_of_neighbors(neighbor);
      for (vertex_t i = 0; i < total_edges; ++i) {
        auto e = i + starting_edge;
        auto n = B.get_edge_weight(e);
        C_col_indices[e] = n;
        math::atomic::add(&C_values[e], weight * B.get_edge_weight(e));
      }

      return false;  // ignored.
    };

    // Perform advance on the above lambda-op
    operators::advance::execute<
        operators::load_balance_t::block_mapped,
        operators::advance_direction_t::forward,  // direction (backward for
                                                  // transpose)
        operators::advance_io_type_t::graph,      // entire graph as input
        operators::advance_io_type_t::none>(      // no output frontier needed
        A, E, spgemm, context);

    /// Step 6. Finally set the sparse-matrix C's data, again.
    C.template set<graph::graph_csr_t<vertex_t, edge_t, weight_t>>(
        A.get_number_of_vertices(), C_nnz, C_row_offsets, C_col_indices,
        C_values);
  }

  /**
   * @brief SpGEMM converges within one iteration.
   *
   * @param context The context of the execution (unused).
   * @return true returns true after one iteration.
   */
  virtual bool is_converged(cuda::multi_context_t& context) {
    return this->iteration == 0 ? false : true;
  }
};  // struct enactor_t

template <typename a_graph_t, typename b_graph_t, typename c_graph_t>
float run(a_graph_t& A,
          b_graph_t& B,
          c_graph_t& C,
          std::shared_ptr<cuda::multi_context_t> context =
              std::shared_ptr<cuda::multi_context_t>(
                  new cuda::multi_context_t(0))  // Context
) {
  using param_type = param_t<a_graph_t, b_graph_t>;
  using result_type = result_t<c_graph_t>;

  using graph_t = a_graph_t;

  param_type param(A, B);
  result_type result(C);

  using problem_type = problem_t<graph_t, param_type, result_type>;
  using enactor_type = enactor_t<problem_type>;

  problem_type problem(A, param, result, context);
  problem.init();
  problem.reset();

  // Disable internal-frontiers:
  enactor_properties_t props;
  props.self_manage_frontiers = true;

  enactor_type enactor(&problem, context, props);
  return enactor.enact();
}

}  // namespace spgemm
}  // namespace gunrock